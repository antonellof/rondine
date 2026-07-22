"""Cluster inventory helpers."""

from __future__ import annotations

from rondine.cluster import ClusterInventory, ClusterNode, plan_cluster_commands


def test_mac_mlx_cluster_plan(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    inv = ClusterInventory(
        name="lab",
        kind="mac",
        nodes=[
            ClusterNode(host="mac1.local", role="head"),
            ClusterNode(host="mac2.local", role="worker"),
        ],
    )
    inv.save()
    lines = plan_cluster_commands(
        inv,
        model_repo="unsloth/Qwen3.6-27B-UD-MLX-4bit",
        engine="mlx",
    )
    assert any("mlx.launch" in line for line in lines)
    assert any("jaccl" in line for line in lines)


def test_spark_cluster_plan_mentions_nccl(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    inv = ClusterInventory(
        name="sparks",
        kind="spark",
        interface="enP2p1s0f1np1",
        nodes=[
            ClusterNode(host="spark-a", role="head"),
            ClusterNode(host="spark-b", role="worker"),
        ],
    )
    lines = plan_cluster_commands(
        inv,
        model_repo="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
        engine="vllm",
    )
    text = "\n".join(lines)
    assert "NCCL" in text or "nccl" in text.lower() or "MN_IF_NAME" in text
    assert "tensor-parallel-size" in text
    assert "Ethernet" in text


def test_rpc_marked_experimental(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("RONDINE_HOME", str(tmp_path))
    inv = ClusterInventory(
        name="rpc",
        kind="mac",
        nodes=[
            ClusterNode(host="a", role="head"),
            ClusterNode(host="b", role="worker"),
        ],
    )
    lines = plan_cluster_commands(
        inv,
        model_repo="unsloth/GLM-5.2-GGUF:UD-IQ2_M",
        engine="llama.cpp",
    )
    assert any("EXPERIMENTAL" in line for line in lines)
