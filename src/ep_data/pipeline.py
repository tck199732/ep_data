"""Workflow orchestration and dynamic parser execution."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from ep_data.config import get_dataset, load_config
from ep_data.diagnostics.template import create_diagnostics_template
from ep_data.downloader import (
    DOWNLOAD_MANIFEST,
    download_submission,
    find_cached_submission,
)
from ep_data.exceptions import ParserLoadError, ProcessingError
from ep_data.utils import (
    atomic_replace_directory,
    fingerprint_file,
    fingerprint_tree,
    read_json,
    write_json_atomic,
)

LOGGER = logging.getLogger(__name__)
ParserFunction = Callable[[Path, Path], None]


def _ensure_diagnostics_template(key: str) -> None:
    try:
        created = create_diagnostics_template(key)
    except (OSError, ValueError) as exc:
        raise ProcessingError(
            f"Could not create diagnostics template for {key!r}: {exc}"
        ) from exc
    if created is not None:
        LOGGER.info("Diagnostics template created: %s", created)


def load_parser(parser_name: str) -> ParserFunction:
    """Import a configured parser and validate its callable entry point."""
    module_name = f"ep_data.parser.{parser_name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ParserLoadError(
                f"Parser module does not exist: {module_name}"
            ) from exc
        raise ParserLoadError(
            f"Parser {module_name} has a missing dependency: {exc.name}"
        ) from exc
    except Exception as exc:
        raise ParserLoadError(f"Could not import parser {module_name}: {exc}") from exc

    parser = getattr(module, "parse", None)
    if not callable(parser):
        raise ParserLoadError(
            f"Parser module {module_name} must expose a callable parse()"
        )
    return parser


def _parser_source(parser: ParserFunction) -> Path:
    source = inspect.getsourcefile(parser)
    if source is None:
        raise ParserLoadError("Could not locate parser source for fingerprinting")
    path = Path(source)
    if not path.is_file():
        raise ParserLoadError(f"Parser source does not exist: {path}")
    return path


def _processing_manifest(data_dir: Path, key: str) -> Path:
    return data_dir / ".ep_data" / "processed" / f"{key}.json"


def _processing_is_current(
    manifest_path: Path,
    output_dir: Path,
    *,
    doi: str,
    parser_name: str,
    raw_fingerprint: str,
    parser_fingerprint: str,
) -> bool:
    if not output_dir.is_dir():
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "completed": True,
        "dataset": output_dir.name,
        "doi": doi,
        "parser": parser_name,
        "parser_fingerprint": parser_fingerprint,
        "source_fingerprint": raw_fingerprint,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def process_dataset(
    key: str,
    config_path: Path,
    data_dir: Path,
    *,
    force: bool = False,
    download_only: bool = False,
    parse_only: bool = False,
) -> None:
    """Download and optionally parse one configured dataset."""
    if download_only and parse_only:
        raise ValueError("download_only and parse_only are mutually exclusive")

    config = get_dataset(load_config(config_path), key)
    data_dir = Path(data_dir)
    raw_root = data_dir / "hep"

    if parse_only:
        raw_data_dir = find_cached_submission(config.doi, raw_root)
    else:
        raw_data_dir = download_submission(config.doi, raw_root, force=force)

    if download_only:
        return
    if config.parser is None:
        LOGGER.info("No parser configured for %s; skipping parsing", key)
        return

    LOGGER.info("Loading parser: %s", config.parser)
    parser = load_parser(config.parser)
    raw_fingerprint = fingerprint_tree(raw_data_dir, excluded_names={DOWNLOAD_MANIFEST})
    parser_fingerprint = fingerprint_file(_parser_source(parser))
    processed_root = data_dir / "processed"
    output_dir = processed_root / key
    manifest_path = _processing_manifest(data_dir, key)

    if not force and _processing_is_current(
        manifest_path,
        output_dir,
        doi=config.doi,
        parser_name=config.parser,
        raw_fingerprint=raw_fingerprint,
        parser_fingerprint=parser_fingerprint,
    ):
        LOGGER.info("Processed data already current: %s", output_dir)
        _ensure_diagnostics_template(key)
        return

    processed_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}-processing-", dir=processed_root))
    try:
        try:
            parser(input_dir=raw_data_dir, output_dir=temporary)
        except Exception as exc:
            raise ProcessingError(f"Parser {config.parser!r} failed: {exc}") from exc
        atomic_replace_directory(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    raw_manifest = read_json(raw_data_dir / DOWNLOAD_MANIFEST)
    write_json_atomic(
        manifest_path,
        {
            "completed": True,
            "dataset": key,
            "doi": config.doi,
            "parser": config.parser,
            "parser_fingerprint": parser_fingerprint,
            "raw_record": raw_manifest.get("record"),
            "raw_version": raw_manifest.get("version"),
            "source_fingerprint": raw_fingerprint,
        },
    )
    _ensure_diagnostics_template(key)
    LOGGER.info("Parser executed successfully: %s", key)
