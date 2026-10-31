import json

import pytest

from ep_data.config import DatasetConfig, get_dataset, load_config
from ep_data.exceptions import ConfigurationError, UnknownDatasetError


def test_load_valid_config(write_config):
    assert load_config(write_config(parser=None)) == {
        "sample": DatasetConfig("10.17182/hepdata.123.v1", None, None)
    }


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "Could not load"),
        ("[]", "top level"),
        (
            json.dumps(
                {
                    "bad/key": {
                        "doi": "10.17182/hepdata.1.v1",
                        "parser": None,
                        "diagnostics": None,
                    }
                }
            ),
            "key",
        ),
        (
            json.dumps({"x": {"parser": None, "diagnostics": None}}),
            "missing required field 'doi'",
        ),
        (
            json.dumps(
                {
                    "x": {
                        "doi": "https://doi.org/10.17182/hepdata.1.v1",
                        "parser": None,
                        "diagnostics": None,
                    }
                }
            ),
            "invalid HEPData DOI",
        ),
        (
            json.dumps(
                {
                    "x": {
                        "doi": "10.17182/hepdata.1.v1",
                        "diagnostics": None,
                    }
                }
            ),
            "missing required field 'parser'",
        ),
        (
            json.dumps({"x": {"doi": "10.17182/hepdata.1.v1", "script": None}}),
            "unsupported legacy field 'script'",
        ),
        (
            json.dumps(
                {
                    "x": {
                        "doi": "10.17182/hepdata.1.v1",
                        "parser": "bad-name",
                        "diagnostics": None,
                    }
                }
            ),
            "invalid parser",
        ),
        (
            json.dumps(
                {
                    "x": {
                        "doi": "10.17182/hepdata.1.v1",
                        "parser": "class",
                        "diagnostics": None,
                    }
                }
            ),
            "invalid parser",
        ),
        (
            json.dumps({"x": {"doi": "10.17182/hepdata.1.v1", "parser": None}}),
            "missing required field 'diagnostics'",
        ),
        (
            json.dumps(
                {
                    "x": {
                        "doi": "10.17182/hepdata.1.v1",
                        "parser": None,
                        "diagnostics": "bad-name",
                    }
                }
            ),
            "invalid diagnostics",
        ),
    ],
)
def test_invalid_config(tmp_path, content, message):
    path = tmp_path / "config.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_config(path)


def test_missing_config(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_config(tmp_path / "missing.json")


def test_unknown_dataset():
    with pytest.raises(UnknownDatasetError, match="Unknown dataset"):
        get_dataset({}, "absent")
