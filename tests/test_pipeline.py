import json
from pathlib import Path

import pytest

from ep_data import pipeline
from ep_data.exceptions import CacheError, ProcessingError

DOI = "10.17182/hepdata.123.v1"


@pytest.fixture(autouse=True)
def isolate_diagnostics_templates(monkeypatch, tmp_path):
    templates = tmp_path / "diagnostics-templates"
    real_create = pipeline.create_diagnostics_template
    monkeypatch.setattr(
        pipeline,
        "create_diagnostics_template",
        lambda key: real_create(key, templates),
    )
    return templates


def test_null_parser_skips_without_processed_dir(
    monkeypatch, write_config, make_raw, tmp_path
):
    config = write_config(parser=None)
    raw = make_raw()
    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)

    pipeline.process_dataset("sample", config, tmp_path / "data")

    assert not (tmp_path / "data" / "processed").exists()


def test_download_only_never_loads_parser(
    monkeypatch, write_config, make_raw, tmp_path
):
    config = write_config(parser="dummy")
    raw = make_raw()
    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)
    monkeypatch.setattr(
        pipeline,
        "load_parser",
        lambda name: pytest.fail("download-only must not load a parser"),
    )

    pipeline.process_dataset("sample", config, tmp_path / "data", download_only=True)


def test_parse_only_uses_cache_without_download(
    monkeypatch, write_config, make_raw, tmp_path
):
    config = write_config(parser=None)
    raw = make_raw()
    monkeypatch.setattr(
        pipeline,
        "download_submission",
        lambda *args, **kwargs: pytest.fail("parse-only accessed downloader"),
    )

    pipeline.process_dataset("sample", config, tmp_path / "data", parse_only=True)
    assert raw.is_dir()


def test_parse_only_missing_cache(write_config, tmp_path):
    config = write_config(parser=None)
    with pytest.raises(CacheError, match="No valid cached"):
        pipeline.process_dataset("sample", config, tmp_path / "data", parse_only=True)


def test_parser_receives_correct_paths_and_writes_output(
    monkeypatch,
    write_config,
    make_raw,
    tmp_path,
    isolate_diagnostics_templates,
):
    config = write_config(parser="dummy")
    raw = make_raw()
    received = {}

    def parser(*, input_dir, output_dir):
        received["input"] = input_dir
        received["output"] = output_dir
        (output_dir / "marker.txt").write_text("called", encoding="utf-8")

    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)
    monkeypatch.setattr(pipeline, "load_parser", lambda name: parser)

    pipeline.process_dataset("sample", config, tmp_path / "data")

    output = tmp_path / "data" / "processed" / "sample"
    assert received["input"] == raw
    assert received["output"] != output
    assert (output / "marker.txt").read_text() == "called"
    manifest = json.loads(
        (tmp_path / "data" / ".ep_data" / "processed" / "sample.json").read_text()
    )
    assert manifest["completed"] is True
    assert manifest["raw_record"] == "ins123"
    assert (isolate_diagnostics_templates / "sample.py").is_file()


def test_processing_cache_and_force(
    monkeypatch,
    write_config,
    make_raw,
    tmp_path,
    isolate_diagnostics_templates,
):
    config = write_config(parser="dummy")
    raw = make_raw()
    calls = []

    def parser(*, input_dir, output_dir):
        calls.append((input_dir, output_dir))
        (output_dir / "count.txt").write_text(str(len(calls)), encoding="utf-8")

    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)
    monkeypatch.setattr(pipeline, "load_parser", lambda name: parser)

    pipeline.process_dataset("sample", config, tmp_path / "data")
    template = isolate_diagnostics_templates / "sample.py"
    assert template.is_file()
    template.unlink()
    pipeline.process_dataset("sample", config, tmp_path / "data")
    assert len(calls) == 1
    assert template.is_file()

    pipeline.process_dataset("sample", config, tmp_path / "data", force=True)
    assert len(calls) == 2
    assert (tmp_path / "data" / "processed" / "sample" / "count.txt").read_text() == "2"


def test_source_change_invalidates_processing_cache(
    monkeypatch, write_config, make_raw, tmp_path
):
    config = write_config(parser="dummy")
    raw = make_raw()
    calls = []

    def parser(*, input_dir, output_dir):
        calls.append(input_dir)
        (output_dir / "marker").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)
    monkeypatch.setattr(pipeline, "load_parser", lambda name: parser)

    pipeline.process_dataset("sample", config, tmp_path / "data")
    (raw / "Table1.yaml").write_text("changed: true\n", encoding="utf-8")
    pipeline.process_dataset("sample", config, tmp_path / "data")
    assert len(calls) == 2


def test_parser_failure_preserves_existing_output(
    monkeypatch,
    write_config,
    make_raw,
    tmp_path,
    isolate_diagnostics_templates,
):
    config = write_config(parser="dummy")
    raw = make_raw()
    output = tmp_path / "data" / "processed" / "sample"
    output.mkdir(parents=True)
    (output / "old.txt").write_text("keep", encoding="utf-8")

    def parser(*, input_dir, output_dir):
        (output_dir / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)
    monkeypatch.setattr(pipeline, "load_parser", lambda name: parser)

    with pytest.raises(ProcessingError, match="boom"):
        pipeline.process_dataset("sample", config, tmp_path / "data")

    assert (output / "old.txt").read_text() == "keep"
    assert not (output / "partial.txt").exists()
    assert not (isolate_diagnostics_templates / "sample.py").exists()


def test_mutually_exclusive_modes(write_config, tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.process_dataset(
            "sample",
            write_config(),
            tmp_path / "data",
            download_only=True,
            parse_only=True,
        )


def test_parser_exception_retains_cause(monkeypatch, write_config, make_raw, tmp_path):
    config = write_config(parser="dummy")
    raw = make_raw()

    def parser(input_dir: Path, output_dir: Path):
        raise LookupError("underlying")

    monkeypatch.setattr(pipeline, "download_submission", lambda *args, **kwargs: raw)
    monkeypatch.setattr(pipeline, "load_parser", lambda name: parser)

    with pytest.raises(ProcessingError) as captured:
        pipeline.process_dataset("sample", config, tmp_path / "data")
    assert isinstance(captured.value.__cause__, LookupError)
