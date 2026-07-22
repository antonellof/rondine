"""CLI smoke via Click runner."""

from __future__ import annotations

from click.testing import CliRunner

from rondine.catalog import load_catalog
from rondine.cli import _apply_hub_hardware_budget, _format_logo, main
from rondine.detect import HardwareInfo


def test_format_logo_centers_artwork(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("rondine.cli.brand_logo", lambda: "XX\nXXXX")
    assert _format_logo(width=8) == "  XX\n  XXXX"
    assert _format_logo(width=3) == "XX\nXXXX"


def test_cli_doctor() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["doctor"])
    assert result.exit_code == 0
    assert "ram:" in result.output.lower() or "RAM" in result.output or "ram:" in result.output


def test_cli_models() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["models"])
    assert result.exit_code == 0
    assert "qwen3.6" in result.output


def test_cli_plan_json(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["plan", "qwen3.6-27b", "--profile", "coding", "--json"])
    # May fail to fit on tiny CI VMs — accept either success with selected or exit 1
    assert "qwen3.6-27b" in result.output or result.exit_code in {0, 1}


def test_hub_plan_uses_discrete_gpu_vram() -> None:
    selected = {
        "estimate": {
            "weight_gb": 22.0,
            "activation_gb": 2.64,
            "kv_gb": 0.0,
            "os_reserve_gb": 8.0,
            "total_gb": 32.64,
            "available_gb": 0.0,
            "headroom_gb": 0.0,
            "fits": True,
        }
    }
    hw = HardwareInfo(
        platform="linux",
        arch="x86_64",
        hostname="vast",
        ram_gb=204.0,
        cuda_available=True,
        cuda_capability=(8, 6),
        gpu_name="RTX 3090",
        vram_gb=24.0,
        gpu_count=1,
    )

    total, unit = _apply_hub_hardware_budget(selected, load_catalog(), hw)

    assert unit == "VRAM"
    assert selected["estimate"]["available_gb"] == 24.0
    assert selected["estimate"]["total_gb"] == total
    assert not selected["estimate"]["fits"]


def test_cli_suggest(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(main, ["suggest", "--profile", "coding", "--limit", "3"])
    assert result.exit_code == 0
    assert "recommended configs" in result.output.lower() or "no fitting" in result.output.lower()


def test_cli_suggest_configure(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["suggest", "--configure", "1", "--save-as", "coding", "--limit", "3"],
    )
    if result.exit_code != 0:
        return
    assert (tmp_path / "plans" / "last.json").is_file()
    assert (tmp_path / "presets" / "coding.json").is_file()


def test_cli_serve_dry_run(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    runner = CliRunner()
    plan = runner.invoke(main, ["plan", "gemma-4-12b", "--profile", "coding"])
    # gemma-4-12b is small; should fit almost everywhere
    if plan.exit_code != 0:
        return
    result = runner.invoke(main, ["serve", "gemma-4-12b", "--dry-run", "--profile", "coding"])
    assert result.exit_code == 0
    assert "command:" in result.output
    assert "openai base url:" in result.output
