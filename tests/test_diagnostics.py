import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
import pytest

from ep_data.diagnostics import runner
from ep_data.diagnostics.loading import load_dataset, load_measurement
from ep_data.diagnostics.style import DEFAULT_MPL_CONFIG, diagnostic_style
from ep_data.diagnostics.template import create_diagnostics_template
from ep_data.exceptions import DiagnosticsError, DiagnosticsLoadError
from ep_data.models import Kinematics, Measurement, Metadata, Qualifier, Xsect
from ep_data.utils import write_json_atomic


def test_load_diagnostics_success(monkeypatch):
    imported = []

    def plot(input_dir, output_dir):
        return (input_dir, output_dir)

    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: imported.append(name) or SimpleNamespace(plot=plot),
    )

    assert runner.load_diagnostics("dummy") is plot
    assert imported == ["ep_data.diagnostics.plots.dummy"]


def test_missing_diagnostics_module(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(runner.importlib, "import_module", missing)

    with pytest.raises(DiagnosticsLoadError, match="does not exist"):
        runner.load_diagnostics("absent")


def test_diagnostics_missing_dependency(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError(name="dependency")

    monkeypatch.setattr(runner.importlib, "import_module", missing)

    with pytest.raises(DiagnosticsLoadError, match="missing dependency"):
        runner.load_diagnostics("dummy")


@pytest.mark.parametrize("value", [None, 42])
def test_missing_or_noncallable_plot(monkeypatch, value):
    monkeypatch.setattr(
        runner.importlib,
        "import_module",
        lambda name: SimpleNamespace(plot=value),
    )

    with pytest.raises(DiagnosticsLoadError, match=r"callable plot\(\)"):
        runner.load_diagnostics("dummy")


def test_diagnostic_style_is_scoped_and_supports_overrides():
    original_dpi = mpl.rcParams["figure.dpi"]
    default_dpi = DEFAULT_MPL_CONFIG["figure.dpi"]

    @diagnostic_style({"figure.dpi": 144})
    def styled():
        assert mpl.rcParams["figure.dpi"] == 144
        assert mpl.rcParams["xtick.direction"] == "in"
        assert mpl.rcParams["ytick.right"] is True

    styled()

    assert mpl.rcParams["figure.dpi"] == original_dpi
    assert DEFAULT_MPL_CONFIG["figure.dpi"] == default_dpi


def test_decorated_plot_fingerprint_uses_dataset_source():
    @diagnostic_style()
    def plot(input_dir, output_dir):
        return (input_dir, output_dir)

    assert runner._source_path(plot).resolve() == Path(__file__).resolve()


def test_diagnostics_template_loads_typed_data_and_has_empty_overrides(tmp_path):
    created = create_diagnostics_template("sample", tmp_path)

    assert created == tmp_path / "sample.py"
    source = created.read_text()
    assert "import matplotlib as mpl" in source
    assert "import matplotlib.pyplot as plt" in source
    assert "import numpy as np" in source
    assert "from ep_data.diagnostics.loading import load_dataset" in source
    assert "STYLE_OVERRIDES: dict[str, Any] = {}" in source
    assert "@diagnostic_style(STYLE_OVERRIDES)" in source
    assert "measurements = load_dataset(input_dir)" in source


def test_diagnostics_template_defaults_to_plots_subdirectory(monkeypatch, tmp_path):
    plots_dir = tmp_path / "diagnostics" / "plots"
    monkeypatch.setattr(
        "ep_data.diagnostics.template.PLOTS_DIR",
        plots_dir,
    )

    assert create_diagnostics_template("sample") == plots_dir / "sample.py"


def test_diagnostics_template_never_overwrites_existing_module(tmp_path):
    destination = tmp_path / "sample.py"
    destination.write_text("# user implementation\n", encoding="utf-8")

    assert create_diagnostics_template("sample", tmp_path) is None
    assert destination.read_text() == "# user implementation\n"


@pytest.mark.parametrize("key", ["bad-key", "class"])
def test_diagnostics_template_rejects_invalid_module_names(tmp_path, key):
    with pytest.raises(ValueError, match="valid Python module"):
        create_diagnostics_template(key, tmp_path)


def test_load_dataset_reconstructs_measurement_dataclasses(tmp_path):
    measurement = Measurement(
        xsect=Xsect(sigU=[1.0, 2.0], sigU_err=[0.1, 0.2]),
        kinematics=Kinematics(Q2=[0.5, 1.0], W=[2.0, 2.1]),
        metadata=Metadata(
            reaction="GAMMA* P --> K+ LAMBDA",
            qualifiers=(Qualifier("W", "2.0-2.1", "GeV"),),
        ),
    )
    write_json_atomic(tmp_path / "measurement.json", measurement)

    loaded = load_dataset(tmp_path)

    assert set(loaded) == {"measurement"}
    assert isinstance(loaded["measurement"], Measurement)
    np.testing.assert_array_equal(loaded["measurement"].xsect.sigU, [1.0, 2.0])
    assert loaded["measurement"].metadata.qualifiers == (
        Qualifier("W", "2.0-2.1", "GeV"),
    )


def test_load_measurement_reports_invalid_documents(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"xsect": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required field"):
        load_measurement(path)


def test_load_dataset_requires_measurement_json(tmp_path):
    with pytest.raises(FileNotFoundError, match="No processed JSON"):
        load_dataset(tmp_path)


def test_null_diagnostics_skips_without_processed_data(write_config, tmp_path):
    config = write_config(diagnostics=None)

    runner.process_diagnostics("sample", config, tmp_path / "data")

    assert not (tmp_path / "data" / "figures").exists()


def test_missing_processed_data_fails_before_loading(
    monkeypatch, write_config, tmp_path
):
    config = write_config(diagnostics="dummy")
    monkeypatch.setattr(
        runner,
        "load_diagnostics",
        lambda name: pytest.fail("loader should not run without processed data"),
    )

    with pytest.raises(DiagnosticsError, match="Processed data is unavailable"):
        runner.process_diagnostics("sample", config, tmp_path / "data")


def test_diagnostics_receives_paths_and_accepts_any_output_format(
    monkeypatch, write_config, tmp_path
):
    config = write_config(diagnostics="dummy")
    input_dir = tmp_path / "data" / "processed" / "sample"
    input_dir.mkdir(parents=True)
    (input_dir / "measurement.json").write_text("{}", encoding="utf-8")
    received = {}

    def plot(*, input_dir, output_dir):
        received["input"] = input_dir
        received["output"] = output_dir
        (output_dir / "overview.svg").write_text("<svg/>", encoding="utf-8")

    monkeypatch.setattr(runner, "load_diagnostics", lambda name: plot)

    runner.process_diagnostics("sample", config, tmp_path / "data")

    output_dir = tmp_path / "data" / "figures" / "sample"
    assert received["input"] == input_dir
    assert received["output"] != output_dir
    assert (output_dir / "overview.svg").is_file()
    manifest = json.loads(
        (tmp_path / "data" / ".ep_data" / "diagnostics" / "sample.json").read_text()
    )
    assert manifest["completed"] is True
    assert manifest["diagnostics"] == "dummy"


def test_diagnostics_cache_and_force(monkeypatch, write_config, tmp_path):
    config = write_config(diagnostics="dummy")
    input_dir = tmp_path / "data" / "processed" / "sample"
    input_dir.mkdir(parents=True)
    (input_dir / "measurement.json").write_text("{}", encoding="utf-8")
    calls = []

    def plot(*, input_dir, output_dir):
        calls.append(input_dir)
        (output_dir / "plot.png").write_bytes(b"figure")

    monkeypatch.setattr(runner, "load_diagnostics", lambda name: plot)

    runner.process_diagnostics("sample", config, tmp_path / "data")
    runner.process_diagnostics("sample", config, tmp_path / "data")
    assert len(calls) == 1

    runner.process_diagnostics("sample", config, tmp_path / "data", force=True)
    assert len(calls) == 2


def test_processed_data_change_invalidates_cache(monkeypatch, write_config, tmp_path):
    config = write_config(diagnostics="dummy")
    input_dir = tmp_path / "data" / "processed" / "sample"
    input_dir.mkdir(parents=True)
    source = input_dir / "measurement.json"
    source.write_text("{}", encoding="utf-8")
    calls = []

    def plot(*, input_dir, output_dir):
        calls.append(input_dir)
        (output_dir / "plot.pdf").write_bytes(b"figure")

    monkeypatch.setattr(runner, "load_diagnostics", lambda name: plot)

    runner.process_diagnostics("sample", config, tmp_path / "data")
    source.write_text('{"changed": true}', encoding="utf-8")
    runner.process_diagnostics("sample", config, tmp_path / "data")

    assert len(calls) == 2


@pytest.mark.parametrize("write_partial", [False, True])
def test_failed_or_empty_diagnostics_preserves_existing_output(
    monkeypatch, write_config, tmp_path, write_partial
):
    config = write_config(diagnostics="dummy")
    input_dir = tmp_path / "data" / "processed" / "sample"
    input_dir.mkdir(parents=True)
    (input_dir / "measurement.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "data" / "figures" / "sample"
    output_dir.mkdir(parents=True)
    (output_dir / "old.png").write_bytes(b"keep")

    def plot(*, input_dir, output_dir):
        if write_partial:
            (output_dir / "partial.png").write_bytes(b"partial")
            raise RuntimeError("boom")

    monkeypatch.setattr(runner, "load_diagnostics", lambda name: plot)

    expected = "boom" if write_partial else "produced no output"
    with pytest.raises(DiagnosticsError, match=expected):
        runner.process_diagnostics("sample", config, tmp_path / "data")

    assert (output_dir / "old.png").read_bytes() == b"keep"
    assert not (output_dir / "partial.png").exists()


def test_cli_lists_without_loading_diagnostics(monkeypatch, write_config, capsys):
    config = write_config(diagnostics="dummy")
    monkeypatch.setattr(
        runner,
        "process_diagnostics",
        lambda *args, **kwargs: pytest.fail("list executed diagnostics"),
    )

    assert runner.main(["--config", str(config), "--list"]) == 0
    assert "sample\tdiagnostics=dummy" in capsys.readouterr().out


def test_cli_isolates_dataset_failures(monkeypatch, tmp_path):
    config = tmp_path / "hep_config.json"
    config.write_text(
        json.dumps(
            {
                key: {
                    "doi": f"10.17182/hepdata.{index}.v1",
                    "parser": None,
                    "diagnostics": "dummy",
                }
                for index, key in enumerate(("one", "two"), start=1)
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def process(key, *args, **kwargs):
        calls.append(key)
        if key == "one":
            raise DiagnosticsError("failed")

    monkeypatch.setattr(runner, "process_diagnostics", process)

    assert runner.main(["--config", str(config)]) == 1
    assert calls == ["one", "two"]
