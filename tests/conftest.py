from __future__ import annotations

import io
import json
import tarfile

import pytest

from ep_data.downloader import DOWNLOAD_MANIFEST


@pytest.fixture
def write_config(tmp_path):
    def _write(
        *,
        parser=None,
        diagnostics=None,
        key="sample",
        doi="10.17182/hepdata.123.v1",
    ):
        path = tmp_path / "hep_config.json"
        path.write_text(
            json.dumps(
                {
                    key: {
                        "doi": doi,
                        "parser": parser,
                        "diagnostics": diagnostics,
                    }
                }
            ),
            encoding="utf-8",
        )
        return path

    return _write


@pytest.fixture
def make_raw(tmp_path):
    def _make(
        *,
        doi="10.17182/hepdata.123.v1",
        record="ins123",
        version=1,
        table_content="independent_variables: []\ndependent_variables: []\n",
    ):
        root = tmp_path / "data" / "hep" / f"HEPData-{record}-v{version}-yaml"
        root.mkdir(parents=True)
        (root / "submission.yaml").write_text(
            "---\ncomment: fixture\n---\ndata_file: Table1.yaml\n",
            encoding="utf-8",
        )
        (root / "Table1.yaml").write_text(table_content, encoding="utf-8")
        (root / DOWNLOAD_MANIFEST).write_text(
            json.dumps(
                {
                    "completed": True,
                    "doi": doi,
                    "record": record,
                    "version": version,
                }
            ),
            encoding="utf-8",
        )
        return root

    return _make


def make_archive(
    files: dict[str, str] | None = None,
    *,
    symlink: tuple[str, str] | None = None,
) -> bytes:
    files = files or {
        "HEPData-ins123-v1-yaml/submission.yaml": (
            "---\ncomment: fixture\n---\ndata_file: Table1.yaml\n"
        ),
        "HEPData-ins123-v1-yaml/Table1.yaml": (
            "independent_variables: []\ndependent_variables: []\n"
        ),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for name, content in files.items():
            encoded = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            bundle.addfile(info, io.BytesIO(encoded))
        if symlink is not None:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            bundle.addfile(info)
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, *, url="", body=b"", error=None):
        self.url = url
        self.body = body
        self.error = error
        self.closed = False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def iter_content(self, chunk_size):
        del chunk_size
        yield self.body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeSession:
    def __init__(self, archive, *, resolution_error=None, download_error=None):
        self.archive = archive
        self.resolution_error = resolution_error
        self.download_error = download_error
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.startswith("https://doi.org/"):
            return FakeResponse(
                url="https://www.hepdata.net/record/ins123?version=1",
                error=self.resolution_error,
            )
        return FakeResponse(body=self.archive, error=self.download_error)

    def close(self):
        self.closed = True
