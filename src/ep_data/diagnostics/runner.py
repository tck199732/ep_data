"""Diagnostics loading, caching, execution, and command-line interface."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from ep_data.config import get_dataset, load_config
from ep_data.exceptions import (
    DiagnosticsError,
    DiagnosticsLoadError,
    HEPDataRetrievalError,
)
from ep_data.utils import (
    atomic_replace_directory,
    fingerprint_file,
    fingerprint_tree,
    read_json,
    write_json_atomic,
)

LOGGER = logging.getLogger("ep_data.diagnostics")
DiagnosticFunction = Callable[[Path, Path], None]
DEFAULT_CONFIG = Path.cwd() / "data" / "hep_config.json"


def load_diagnostics(name: str) -> DiagnosticFunction:
    """Import a configured diagnostics module and validate its plot function."""
    module_name = f"ep_data.diagnostics.plots.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise DiagnosticsLoadError(
                f"Diagnostics module does not exist: {module_name}"
            ) from exc
        raise DiagnosticsLoadError(
            f"Diagnostics module {module_name} has a missing dependency: {exc.name}"
        ) from exc
    except Exception as exc:
        raise DiagnosticsLoadError(
            f"Could not import diagnostics module {module_name}: {exc}"
        ) from exc

    plot = getattr(module, "plot", None)
    if not callable(plot):
        raise DiagnosticsLoadError(
            f"Diagnostics module {module_name} must expose a callable plot()"
        )
    return plot


def _source_path(plot: DiagnosticFunction) -> Path:
    source = inspect.getsourcefile(inspect.unwrap(plot))
    if source is None:
        raise DiagnosticsLoadError(
            "Could not locate diagnostics source for fingerprinting"
        )
    path = Path(source)
    if not path.is_file():
        raise DiagnosticsLoadError(f"Diagnostics source does not exist: {path}")
    return path


def _diagnostics_fingerprint(plot: DiagnosticFunction) -> str:
    style_source = Path(__file__).with_name("style.py")
    digest = hashlib.sha256()
    for source in (_source_path(plot), style_source):
        digest.update(fingerprint_file(source).encode("ascii"))
    return digest.hexdigest()


def _manifest_path(data_dir: Path, key: str) -> Path:
    return data_dir / ".ep_data" / "diagnostics" / f"{key}.json"


def _has_output(directory: Path) -> bool:
    return directory.is_dir() and any(path.is_file() for path in directory.rglob("*"))


def _is_current(
    manifest_path: Path,
    output_dir: Path,
    *,
    key: str,
    diagnostics_name: str,
    input_fingerprint: str,
    diagnostics_fingerprint: str,
) -> bool:
    if not _has_output(output_dir):
        return False
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = {
        "completed": True,
        "dataset": key,
        "diagnostics": diagnostics_name,
        "diagnostics_fingerprint": diagnostics_fingerprint,
        "input_fingerprint": input_fingerprint,
        "output_fingerprint": fingerprint_tree(output_dir),
    }
    return all(manifest.get(field) == value for field, value in expected.items())


def process_diagnostics(
    key: str,
    config_path: Path,
    data_dir: Path,
    *,
    force: bool = False,
) -> None:
    """Generate cached diagnostic figures for one processed dataset."""
    config = get_dataset(load_config(config_path), key)
    if config.diagnostics is None:
        LOGGER.info("No diagnostics configured for %s; skipping", key)
        return

    data_dir = Path(data_dir)
    input_dir = data_dir / "processed" / key
    if not input_dir.is_dir():
        raise DiagnosticsError(
            f"Processed data is unavailable for {key!r}: {input_dir}"
        )

    LOGGER.info("Loading diagnostics: %s", config.diagnostics)
    plot = load_diagnostics(config.diagnostics)
    input_fingerprint = fingerprint_tree(input_dir)
    diagnostics_fingerprint = _diagnostics_fingerprint(plot)
    figures_root = data_dir / "figures"
    output_dir = figures_root / key
    manifest_path = _manifest_path(data_dir, key)

    if not force and _is_current(
        manifest_path,
        output_dir,
        key=key,
        diagnostics_name=config.diagnostics,
        input_fingerprint=input_fingerprint,
        diagnostics_fingerprint=diagnostics_fingerprint,
    ):
        LOGGER.info("Diagnostic figures already current: %s", output_dir)
        return

    figures_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}-diagnostics-", dir=figures_root))
    try:
        try:
            plot(input_dir=input_dir, output_dir=temporary)
        except Exception as exc:
            raise DiagnosticsError(
                f"Diagnostics {config.diagnostics!r} failed: {exc}"
            ) from exc
        if not _has_output(temporary):
            raise DiagnosticsError(
                f"Diagnostics {config.diagnostics!r} produced no output files"
            )
        atomic_replace_directory(temporary, output_dir)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    write_json_atomic(
        manifest_path,
        {
            "completed": True,
            "dataset": key,
            "diagnostics": config.diagnostics,
            "diagnostics_fingerprint": diagnostics_fingerprint,
            "input_fingerprint": input_fingerprint,
            "output_fingerprint": fingerprint_tree(output_dir),
        },
    )
    LOGGER.info("Diagnostic figures generated: %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots from processed datasets."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        metavar="KEY",
        help="plot one dataset (repeat to select multiple datasets)",
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate diagnostic outputs"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="list configured diagnostics",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"dataset registry (default: {DEFAULT_CONFIG})",
    )
    return parser


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        datasets = load_config(args.config)
    except HEPDataRetrievalError as exc:
        LOGGER.error("%s", exc)
        return 1

    if args.list_datasets:
        for key, dataset in datasets.items():
            name = dataset.diagnostics if dataset.diagnostics is not None else "none"
            print(f"{key}\tdiagnostics={name}")
        return 0

    selected = _unique(args.datasets) if args.datasets else list(datasets)
    data_dir = args.config.resolve().parent
    failures: list[tuple[str, str]] = []
    for key in selected:
        LOGGER.info("Processing diagnostics for %s", key)
        try:
            process_diagnostics(key, args.config, data_dir, force=args.force)
        except Exception as exc:
            failures.append((key, str(exc)))
            LOGGER.error("Failed %s: %s", key, exc)

    if failures:
        LOGGER.error("%d dataset diagnostic(s) failed", len(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
