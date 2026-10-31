import io
import json
import tarfile

import pytest
import requests
from conftest import FakeSession, make_archive

from ep_data.downloader import hepdata as downloader
from ep_data.exceptions import CacheError, DownloadError, UnsafeArchiveError

DOI = "10.17182/hepdata.123.v1"


def test_successful_download(monkeypatch, tmp_path):
    session = FakeSession(make_archive())
    monkeypatch.setattr(downloader, "_build_session", lambda: session)

    result = downloader.download_submission(DOI, tmp_path / "hep")

    assert result.name == "HEPData-ins123-v1-yaml"
    assert (result / "submission.yaml").is_file()
    assert (result / "Table1.yaml").is_file()
    manifest = json.loads((result / downloader.DOWNLOAD_MANIFEST).read_text())
    assert manifest["completed"] is True
    assert manifest["record"] == "ins123"
    assert len(session.calls) == 2
    assert session.calls[1][1]["stream"] is True
    assert session.closed


def test_cache_reuse_without_network(monkeypatch, make_raw, tmp_path):
    cached = make_raw()
    monkeypatch.setattr(
        downloader,
        "_build_session",
        lambda: pytest.fail("cache reuse must not create a network session"),
    )
    assert downloader.download_submission(DOI, tmp_path / "data" / "hep") == cached


def test_force_redownloads(monkeypatch, make_raw, tmp_path):
    cached = make_raw(table_content="old: true\n")
    session = FakeSession(make_archive())
    monkeypatch.setattr(downloader, "_build_session", lambda: session)

    result = downloader.download_submission(DOI, tmp_path / "data" / "hep", force=True)

    assert result == cached
    assert "old: true" not in (result / "Table1.yaml").read_text()
    assert len(session.calls) == 2


def test_failed_force_preserves_valid_cache(monkeypatch, make_raw, tmp_path):
    cached = make_raw(table_content="old: true\n")
    session = FakeSession(b"not a tar")
    monkeypatch.setattr(downloader, "_build_session", lambda: session)

    with pytest.raises(DownloadError):
        downloader.download_submission(DOI, tmp_path / "data" / "hep", force=True)

    assert (cached / "Table1.yaml").read_text() == "old: true\n"
    assert downloader.find_cached_submission(DOI, tmp_path / "data" / "hep") == cached


def test_resolution_http_error(monkeypatch, tmp_path):
    error = requests.HTTPError("not found")
    monkeypatch.setattr(
        downloader,
        "_build_session",
        lambda: FakeSession(make_archive(), resolution_error=error),
    )
    with pytest.raises(DownloadError, match="resolve"):
        downloader.download_submission(DOI, tmp_path / "hep")


def test_download_http_error(monkeypatch, tmp_path):
    error = requests.HTTPError("server error")
    monkeypatch.setattr(
        downloader,
        "_build_session",
        lambda: FakeSession(make_archive(), download_error=error),
    )
    with pytest.raises(DownloadError, match="download"):
        downloader.download_submission(DOI, tmp_path / "hep")


@pytest.mark.parametrize("member", ["../escape.yaml", "/absolute.yaml"])
def test_archive_path_traversal_rejected(monkeypatch, tmp_path, member):
    session = FakeSession(make_archive({member: "bad"}))
    monkeypatch.setattr(downloader, "_build_session", lambda: session)
    with pytest.raises(UnsafeArchiveError):
        downloader.download_submission(DOI, tmp_path / "hep")


def test_archive_symlink_rejected(monkeypatch, tmp_path):
    session = FakeSession(make_archive(symlink=("unsafe", "../target")))
    monkeypatch.setattr(downloader, "_build_session", lambda: session)
    with pytest.raises(UnsafeArchiveError):
        downloader.download_submission(DOI, tmp_path / "hep")


def test_missing_referenced_table(monkeypatch, tmp_path):
    archive = make_archive(
        {
            "root/submission.yaml": "---\ndata_file: missing.yaml\n",
        }
    )
    monkeypatch.setattr(downloader, "_build_session", lambda: FakeSession(archive))
    with pytest.raises(CacheError, match="Missing referenced"):
        downloader.download_submission(DOI, tmp_path / "hep")


def test_invalid_yaml_table(monkeypatch, tmp_path):
    archive = make_archive(
        {
            "root/submission.yaml": "---\ndata_file: Table1.yaml\n",
            "root/Table1.yaml": "invalid: [",
        }
    )
    monkeypatch.setattr(downloader, "_build_session", lambda: FakeSession(archive))
    with pytest.raises(CacheError, match="Invalid referenced"):
        downloader.download_submission(DOI, tmp_path / "hep")


def test_ambiguous_archive_layout(monkeypatch, tmp_path):
    archive = make_archive(
        {
            "one/submission.yaml": "---\ndata_file: Table.yaml\n",
            "one/Table.yaml": "{}\n",
            "two/submission.yaml": "---\ndata_file: Table.yaml\n",
            "two/Table.yaml": "{}\n",
        }
    )
    monkeypatch.setattr(downloader, "_build_session", lambda: FakeSession(archive))
    with pytest.raises(DownloadError, match="exactly one"):
        downloader.download_submission(DOI, tmp_path / "hep")


def test_retry_policy():
    session = downloader._build_session()
    try:
        retries = session.get_adapter("https://").max_retries
        assert retries.total == 3
        assert 429 in retries.status_forcelist
        assert retries.respect_retry_after_header
    finally:
        session.close()


def test_incomplete_cache_is_rejected(tmp_path):
    root = tmp_path / "HEPData-ins123-v1-yaml"
    root.mkdir()
    (root / downloader.DOWNLOAD_MANIFEST).write_text(
        json.dumps({"completed": True, "doi": DOI}), encoding="utf-8"
    )
    with pytest.raises(CacheError, match="No valid cached"):
        downloader.find_cached_submission(DOI, tmp_path)


def test_device_archive_member_rejected(monkeypatch, tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        info = tarfile.TarInfo("device")
        info.type = tarfile.CHRTYPE
        bundle.addfile(info)
    monkeypatch.setattr(
        downloader,
        "_build_session",
        lambda: FakeSession(buffer.getvalue()),
    )
    with pytest.raises(UnsafeArchiveError):
        downloader.download_submission(DOI, tmp_path / "hep")
