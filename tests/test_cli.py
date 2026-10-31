import json

from ep_data import cli


def make_config(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "one": {
                    "doi": "10.17182/hepdata.1.v1",
                    "parser": None,
                    "diagnostics": None,
                },
                "two": {
                    "doi": "10.17182/hepdata.2.v1",
                    "parser": None,
                    "diagnostics": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return config


def test_list_has_no_side_effects(monkeypatch, tmp_path, capsys):
    config = make_config(tmp_path)
    monkeypatch.setattr(
        cli,
        "process_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("list executed pipeline")
        ),
    )
    assert cli.main(["--config", str(config), "--list"]) == 0
    output = capsys.readouterr().out
    assert "one" in output
    assert "two" in output


def test_multiple_datasets_and_failure_isolation(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    calls = []

    def process(key, *args, **kwargs):
        calls.append(key)
        if key == "one":
            raise RuntimeError("failed")

    monkeypatch.setattr(cli, "process_dataset", process)
    result = cli.main(
        [
            "--config",
            str(config),
            "--dataset",
            "one",
            "--dataset",
            "two",
        ]
    )
    assert result == 1
    assert calls == ["one", "two"]


def test_no_selection_processes_all(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    calls = []
    monkeypatch.setattr(
        cli,
        "process_dataset",
        lambda key, *args, **kwargs: calls.append(key),
    )
    assert cli.main(["--config", str(config)]) == 0
    assert calls == ["one", "two"]


def test_unknown_dataset_returns_failure(monkeypatch, tmp_path):
    config = make_config(tmp_path)
    assert (
        cli.main(["--config", str(config), "--dataset", "absent", "--parse-only"]) == 1
    )


def test_mutually_exclusive_cli_modes():
    parser = cli.build_parser()
    try:
        parser.parse_args(["--download-only", "--parse-only"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("mutually exclusive modes were accepted")
