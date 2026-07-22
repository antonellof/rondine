# Clusters (0.2)

Rondine only wraps **upstream** multi-node mechanisms. It does not invent a scheduler.

## Requirements

- Homogeneous nodes (same platform / memory class)
- Passwordless SSH
- Matching engine versions on every node
- Private high-speed fabric (Thunderbolt / RDMA / RoCE) — never silent Ethernet fallback
- Dual-node is the validated target; larger inventories are diagnostic-only

## Mac (MLX)

```bash
rondine cluster init lab --kind mac --head mac1.local --worker mac2.local
rondine cluster doctor lab
rondine cluster plan lab --engine mlx --model-repo unsloth/Qwen3.6-27B-UD-MLX-4bit
```

Uses `mlx.launch --backend jaccl --hostfile …`.

## Mac GGUF (experimental)

`llama.cpp` RPC (`ggml-rpc-server` + `--rpc`) is **trusted-LAN only**. Do not expose it.

## DGX Spark

```bash
rondine cluster init sparks --kind spark --head spark-a --worker spark-b --interface enP2p1s0f1np1
rondine cluster doctor sparks
rondine cluster plan sparks --engine vllm --model-repo unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
```

Follow NVIDIA connect-two-sparks + Ray multi-node serving. Pin NCCL to ConnectX RoCE interfaces before serving.
