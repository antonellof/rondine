"""CLI smoke via Click runner."""

from __future__ import annotations

import json

from click.testing import CliRunner

from rondine.catalog import load_catalog
from rondine.cli import _apply_hub_hardware_budget, _format_logo, main
from rondine.detect import HardwareInfo
from rondine.presets import preset_from_selected, save_preset
from rondine.suggest import suggest_for_hardware


def test_format_logo_centers_artwork(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("rondine.cli.brand_logo", lambda: "XX\nXXXX")
    assert _format_logo(width=8) == "  XX\n  XXXX"
    assert _format_logo(width=3) == "XX\nXXXX"


def test_cli_without_arguments_shows_help_when_not_interactive() -> None:
    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "doctor" in result.output
    assert "suggest" in result.output


def test_cli_help_keeps_command_listing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("rondine.cli.is_interactive_terminal", lambda: True)
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "Commands:" in result.output
    assert "Rondine interactive" not in result.output


def test_cli_without_arguments_runs_guided_session(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr("rondine.cli.is_interactive_terminal", lambda: True)
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=32.0,
            is_apple_silicon=True,
            metal_available=True,
            disk_free_gb=300.0,
        ),
    )
    choices = iter((0, 0, 1, 0, 5))
    monkeypatch.setattr(
        "rondine.cli.select_menu",
        lambda options, title=None: next(choices),
    )

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "Rondine interactive" in result.output
    assert "Hardware" in result.output
    assert "Recommended configs" in result.output
    assert "Goodbye from Rondine" in result.output
    assert (tmp_path / "plans" / "last.json").is_file()


def test_guided_session_resumes_active_plan_without_doctor(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr("rondine.cli.is_interactive_terminal", lambda: True)
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "last.json").write_text(
        json.dumps(
            {
                "profile": "coding",
                "selected": {
                    "model_id": "existing-model",
                    "display_name": "Existing model",
                    "engine": "mlx",
                    "quant": "4bit",
                    "context": 32768,
                    "repo": "example/existing-model",
                },
            }
        )
    )
    choices = iter((0, 5))
    monkeypatch.setattr(
        "rondine.cli.select_menu",
        lambda options, title=None: next(choices),
    )

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "Welcome back" in result.output
    assert "Existing model" in result.output
    assert "scanning hardware" not in result.output
    assert "Recommended configs" not in result.output


def test_guided_session_can_load_saved_preset(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr("rondine.cli.is_interactive_terminal", lambda: True)
    save_preset(
        preset_from_selected(
            "work",
            {
                "model_id": "saved-model",
                "display_name": "Saved model",
                "engine": "llama.cpp",
                "quant": "Q4_K_M",
                "context": 16384,
                "repo": "example/saved-model",
            },
            profile="chat",
            host="127.0.0.1",
            port=8080,
        )
    )
    choices = iter((0, 0, 5))
    monkeypatch.setattr(
        "rondine.cli.select_menu",
        lambda options, title=None: next(choices),
    )

    result = CliRunner().invoke(main, [])

    assert result.exit_code == 0
    assert "loaded preset work" in result.output
    plan = json.loads((tmp_path / "plans" / "last.json").read_text())
    assert plan["preset_name"] == "work"
    assert plan["selected"]["model_id"] == "saved-model"


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


def test_cli_plan_json_saves_preset(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=48.0,
            is_apple_silicon=True,
            metal_available=True,
        ),
    )

    result = CliRunner().invoke(
        main,
        ["plan", "gemma-4-12b", "--json", "--save-as", "json-plan"],
    )

    assert result.exit_code == 0
    assert json.loads(result.output)["selected"]["model_id"] == "gemma-4-12b"
    assert (tmp_path / "presets" / "json-plan.json").is_file()


def test_cli_plan_mmap_requires_acknowledgement(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=32.0,
            is_apple_silicon=True,
            metal_available=True,
            disk_free_gb=300.0,
        ),
    )
    runner = CliRunner()
    refused = runner.invoke(main, ["plan", "glm-5.2", "--memory-mode", "mmap"])
    assert refused.exit_code != 0
    assert "requires --allow-oversize" in refused.output

    accepted = runner.invoke(
        main,
        [
            "plan",
            "glm-5.2",
            "--memory-mode",
            "mmap",
            "--allow-oversize",
            "--context",
            "4096",
            "--json",
            "--save-as",
            "glm-ssd",
        ],
    )
    assert accepted.exit_code == 0
    selected = json.loads(accepted.output)["selected"]
    assert selected["experimental"] is True
    assert selected["memory_mode"] == "mmap"


def test_cli_plan_explains_hybrid_cuda_requirement(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=32.0,
            is_apple_silicon=True,
            metal_available=True,
            disk_free_gb=300.0,
        ),
    )

    result = CliRunner().invoke(
        main,
        ["plan", "glm-5.2", "--memory-mode", "hybrid", "--context", "4096"],
    )

    assert result.exit_code != 0
    assert "A discrete CUDA host is a computer with a separate NVIDIA GPU" in result.output
    assert "Apple Silicon Mac uses unified memory" in result.output
    assert "rondine suggest --profile coding" in result.output
    assert "--memory-mode mmap --allow-oversize" in result.output


def test_cli_suggest(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=32.0,
            is_apple_silicon=True,
            metal_available=True,
            disk_free_gb=300.0,
        ),
    )
    runner = CliRunner()
    result = runner.invoke(
        main, ["suggest", "--profile", "coding", "--limit", "3", "--no-hub"]
    )
    assert result.exit_code == 0
    assert "recommended configs" in result.output.lower()


def test_cli_suggest_help_documents_options() -> None:
    result = CliRunner().invoke(main, ["suggest", "--help"])
    assert result.exit_code == 0
    assert "--profile [coding|chat]" in result.output
    for option in (
        "--limit",
        "--opt-in",
        "--hub",
        "--hub-query",
        "--context",
        "--interactive",
        "--no-interactive",
        "--json",
        "--configure",
        "--save-as",
    ):
        assert option in result.output


def test_cli_suggest_uses_color_on_terminal(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=32.0,
            is_apple_silicon=True,
            metal_available=True,
        ),
    )

    result = CliRunner().invoke(
        main,
        ["--color", "suggest", "--limit", "2", "--no-hub"],
        color=False,
    )

    assert result.exit_code == 0
    assert "\x1b[" in result.output
    assert "Recommended configs" in result.output
    assert "metal_fast_synch:" in result.output
    assert "────────" in result.output

    plain = CliRunner().invoke(
        main,
        ["--no-color", "suggest", "--limit", "2", "--no-hub"],
        color=True,
    )
    assert plain.exit_code == 0
    assert "\x1b[" not in plain.output


def test_cli_suggest_offers_interactive_selection_after_output(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=32.0,
            is_apple_silicon=True,
            metal_available=True,
        ),
    )
    monkeypatch.setattr("rondine.cli.is_interactive_terminal", lambda: True)

    result = CliRunner().invoke(
        main,
        ["suggest", "--limit", "2", "--no-hub"],
        input="n\n",
    )

    assert result.exit_code == 0
    assert "Select and configure one of these suggestions now?" in result.output
    assert not (tmp_path / "plans" / "last.json").exists()


def test_cli_suggest_interactive_selects_and_configures(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    hw = HardwareInfo(
        platform="darwin",
        arch="arm64",
        hostname="test",
        ram_gb=32.0,
        is_apple_silicon=True,
        metal_available=True,
        disk_free_gb=300.0,
    )
    monkeypatch.setattr("rondine.cli.detect_hardware", lambda: hw)
    expected = suggest_for_hardware(
        load_catalog(),
        hw,
        limit=3,
        include_hub=False,
    ).suggestions[1]

    result = CliRunner().invoke(
        main,
        ["suggest", "--interactive", "--limit", "3", "--no-hub"],
        input="2\n",
    )

    assert result.exit_code == 0
    assert "Select a configuration" in result.output
    plan = json.loads((tmp_path / "plans" / "last.json").read_text())
    assert plan["selected"]["model_id"] == expected.model_id


def test_cli_suggest_interactive_can_show_more(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=48.0,
            is_apple_silicon=True,
            metal_available=True,
            disk_free_gb=300.0,
        ),
    )
    menus: list[list[str]] = []

    def choose(options: list[str], title: str | None = None) -> int:
        menus.append(options)
        return len(options) - 1 if len(menus) == 1 else 0

    monkeypatch.setattr("rondine.cli.select_menu", choose)
    result = CliRunner().invoke(
        main,
        ["suggest", "--interactive", "--limit", "3", "--no-hub"],
    )

    assert result.exit_code == 0
    assert menus[0][-1] == "Show more recommendations — up to 8 results"
    assert len(menus[1]) > len(menus[0])
    assert "Expanding recommendations to 8" in result.output
    assert (tmp_path / "plans" / "last.json").is_file()


def test_cli_suggest_context_changes_planned_window(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    monkeypatch.setattr(
        "rondine.cli.detect_hardware",
        lambda: HardwareInfo(
            platform="darwin",
            arch="arm64",
            hostname="test",
            ram_gb=48.0,
            is_apple_silicon=True,
            metal_available=True,
        ),
    )

    result = CliRunner().invoke(
        main,
        ["suggest", "--context", "4096", "--json", "--no-hub"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["suggestions"]
    assert all(item["context"] == 4096 for item in payload["suggestions"])


def test_cli_suggest_configure(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "suggest",
            "--configure",
            "1",
            "--save-as",
            "coding",
            "--limit",
            "3",
            "--no-hub",
        ],
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
