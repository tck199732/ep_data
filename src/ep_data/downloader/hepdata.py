"""HEPData DOI resolution, YAML archive download, and raw-data caching."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlparse

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ep_data.exceptions import (
    CacheError,
    DownloadError,
    UnsafeArchiveError,
)
from ep_data.utils import atomic_replace_directory, read_json

LOGGER = logging.getLogger(__name__)
DOWNLOAD_MANIFEST = ".hepdata-retrieval-download.json"
TIMEOUT = (10, 60)
RECORD_PATTERN = re.compile(r"^/record/(ins\d+)$")


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    return session


def _resolve_doi(doi: str, session: requests.Session) -> tuple[str, int]:
    url = f"https://doi.org/{doi}"
    try:
        response = session.get(url, timeout=TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Could not resolve HEPData DOI {doi!r}: {exc}") from exc

    try:
        parsed = urlparse(response.url)
        match = RECORD_PATTERN.fullmatch(parsed.path.rstrip("/"))
        versions = parse_qs(parsed.query).get("version", [])
        if match is None or len(versions) != 1:
            raise DownloadError(
                f"DOI {doi!r} did not resolve to a versioned HEPData record: "
                f"{response.url}"
            )
        try:
            version = int(versions[0])
        except ValueError as exc:
            raise DownloadError(
                f"DOI {doi!r} resolved with an invalid version: {versions[0]!r}"
            ) from exc
        if version < 1:
            raise DownloadError(f"DOI {doi!r} resolved with invalid version {version}")
        return match.group(1), version
    finally:
        response.close()


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise UnsafeArchiveError(f"Unsafe archive member path: {name!r}")
    return path


def _extract_archive(archive: Path, destination: Path) -> Path:
    try:
        with tarfile.open(archive, mode="r:*") as bundle:
            for member in bundle.getmembers():
                relative = _safe_member_path(member.name)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = bundle.extractfile(member)
                    if source is None:
                        raise DownloadError(
                            f"Could not read archive member {member.name!r}"
                        )
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                else:
                    raise UnsafeArchiveError(
                        f"Unsupported archive member type: {member.name!r}"
                    )
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, UnsafeArchiveError):
            raise
        raise DownloadError(f"Could not extract YAML archive: {exc}") from exc

    submissions = list(destination.rglob("submission.yaml"))
    if len(submissions) != 1:
        raise DownloadError(
            "YAML archive must contain exactly one submission.yaml; "
            f"found {len(submissions)}"
        )
    dataset_root = submissions[0].parent
    if dataset_root != destination:
        outside = [
            path
            for path in destination.iterdir()
            if path != dataset_root and path.name != "__MACOSX"
        ]
        if outside:
            raise DownloadError("YAML archive has an ambiguous top-level layout")
    return dataset_root


def _local_reference(root: Path, raw_name: object) -> Path:
    if not isinstance(raw_name, str) or not raw_name:
        raise CacheError(f"Invalid data_file reference: {raw_name!r}")
    relative = PurePosixPath(raw_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise CacheError(f"Unsafe data_file reference: {raw_name!r}")
    path = root.joinpath(*relative.parts)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise CacheError(f"Referenced data file is not YAML: {raw_name!r}")
    return path


def validate_submission(root: Path) -> list[Path]:
    """Validate submission YAML and all locally referenced YAML table files."""
    root = Path(root)
    submission = root / "submission.yaml"
    if not submission.is_file():
        raise CacheError(f"Missing submission.yaml in {root}")

    try:
        with submission.open(encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except (OSError, yaml.YAMLError) as exc:
        raise CacheError(f"Invalid submission.yaml in {root}: {exc}") from exc

    tables: list[Path] = []
    for document in documents:
        if not isinstance(document, dict) or "data_file" not in document:
            continue
        table = _local_reference(root, document["data_file"])
        if not table.is_file():
            raise CacheError(f"Missing referenced YAML table: {table.name}")
        try:
            with table.open(encoding="utf-8") as stream:
                yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise CacheError(
                f"Invalid referenced YAML table {table.name}: {exc}"
            ) from exc
        tables.append(table)
    return tables


def _valid_cached_path(path: Path, doi: str) -> bool:
    try:
        manifest = read_json(path / DOWNLOAD_MANIFEST)
        if manifest.get("completed") is not True or manifest.get("doi") != doi:
            return False
        validate_submission(path)
    except (OSError, ValueError, json.JSONDecodeError, CacheError):
        return False
    return True


def find_cached_submission(doi: str, raw_root: Path) -> Path:
    """Find a complete cached record without making a network request."""
    raw_root = Path(raw_root)
    candidates = sorted(raw_root.glob("HEPData-ins*-v*-yaml"))
    for candidate in candidates:
        if candidate.is_dir() and _valid_cached_path(candidate, doi):
            return candidate
    raise CacheError(f"No valid cached raw data found for {doi}")


def _stream_archive(url: str, destination: Path, session: requests.Session) -> str:
    digest = hashlib.sha256()
    try:
        with session.get(url, timeout=TIMEOUT, stream=True) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        digest.update(chunk)
    except (requests.RequestException, OSError) as exc:
        raise DownloadError(f"Could not download HEPData YAML archive: {exc}") from exc
    return digest.hexdigest()


def download_submission(
    doi: str,
    raw_root: Path,
    *,
    force: bool = False,
) -> Path:
    """Download and cache a complete HEPData YAML submission."""
    raw_root = Path(raw_root)
    if not force:
        try:
            cached = find_cached_submission(doi, raw_root)
        except CacheError:
            pass
        else:
            LOGGER.info("Raw data already cached: %s", cached)
            return cached

    raw_root.mkdir(parents=True, exist_ok=True)
    session = _build_session()
    try:
        record, version = _resolve_doi(doi, session)
        target = raw_root / f"HEPData-{record}-v{version}-yaml"
        if not force and _valid_cached_path(target, doi):
            return target

        source_url = (
            f"https://www.hepdata.net/download/submission/{record}/{version}/yaml"
        )
        LOGGER.info("Downloading HEPData submission: %s", source_url)
        with tempfile.TemporaryDirectory(
            prefix=".hepdata-download-", dir=raw_root
        ) as temporary_name:
            temporary = Path(temporary_name)
            archive = temporary / "submission.tar"
            archive_sha256 = _stream_archive(source_url, archive, session)
            extracted = temporary / "extracted"
            extracted.mkdir()
            dataset_root = _extract_archive(archive, extracted)
            validate_submission(dataset_root)
            manifest = {
                "archive_sha256": archive_sha256,
                "completed": True,
                "doi": doi,
                "record": record,
                "source_url": source_url,
                "version": version,
            }
            with (dataset_root / DOWNLOAD_MANIFEST).open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
            atomic_replace_directory(dataset_root, target)
        LOGGER.info("Raw data saved: %s", target)
        return target
    finally:
        session.close()
