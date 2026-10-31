from types import SimpleNamespace

import pytest

from ep_data import pipeline
from ep_data.exceptions import ParserLoadError


def test_load_parser_success(monkeypatch):
    def parse(input_dir, output_dir):
        return (input_dir, output_dir)

    monkeypatch.setattr(
        pipeline.importlib,
        "import_module",
        lambda name: SimpleNamespace(parse=parse),
    )
    assert pipeline.load_parser("dummy") is parse


def test_missing_parser_module(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(pipeline.importlib, "import_module", missing)
    with pytest.raises(ParserLoadError, match="does not exist"):
        pipeline.load_parser("absent")


def test_parser_missing_dependency(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name="dependency")

    monkeypatch.setattr(pipeline.importlib, "import_module", missing)
    with pytest.raises(ParserLoadError, match="missing dependency"):
        pipeline.load_parser("dummy")


@pytest.mark.parametrize("value", [None, 42])
def test_missing_or_noncallable_parse(monkeypatch, value):
    monkeypatch.setattr(
        pipeline.importlib,
        "import_module",
        lambda name: SimpleNamespace(parse=value),
    )
    with pytest.raises(ParserLoadError, match="callable parse"):
        pipeline.load_parser("dummy")
