"""Hugging Face Hub discovery — complements the curated catalog."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rondine.catalog import Catalog
from rondine.detect import HardwareInfo
from rondine.planner import available_memory_gb

EngineHint = Literal["llama.cpp", "mlx", "vllm", "unknown"]

# Prefer these publishers when ranking Hub search hits for local serving.
PREFERRED_ORGS = (
    "ggml-org",
    "bartowski",
    "mlx-community",
    "Qwen",
    "google",
    "deepseek-ai",
    "zai-org",
    "lmstudio-community",
    "unsloth",
    "mudler",
)

QUANT_PATTERNS = (
    r"(UD-Q\d+_K_XL)",
    r"(UD-IQ\d+_[A-Z]+)",
    r"(Q\d+_K_[A-Z]+)",
    r"(IQ\d+_[A-Z]+)",
    r"(Q\d+_0)",
    r"(Q\d+_1)",
    r"(BF16)",
    r"(FP16)",
    r"(NVFP4)",
    r"(\d+bit)",
    r"(4bit)",
    r"(8bit)",
    r"(mxfp4)",
)


@dataclass
class HubFile:
    path: str
    size_gb: float
    quant: str | None = None
    is_mmproj: bool = False
    is_gguf: bool = False


@dataclass
class HubModelHit:
    repo_id: str
    downloads: int = 0
    likes: int = 0
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    engine_hint: EngineHint = "unknown"
    format_hint: str = "unknown"
    curated: bool = False
    score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HubInspectResult:
    repo_id: str
    engine_hint: EngineHint
    format_hint: str
    files: list[HubFile]
    recommended_quant: str | None
    recommended_file: str | None
    weight_gb: float
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    notes: list[str] = field(default_factory=list)

    def to_plan_selected(
        self,
        *,
        model_id: str | None = None,
        profile: str = "coding",
        context: int = 32768,
        sampling: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a planner-compatible selected payload for pull/serve."""
        mid = model_id or self.repo_id.replace("/", "--").lower()
        include = []
        if self.recommended_quant:
            include.append(f"*{self.recommended_quant}*")
        elif self.recommended_file:
            include.append(self.recommended_file)
        else:
            include.append("*.gguf" if self.format_hint == "gguf" else "*")
        mmproj = next((f.path for f in self.files if f.is_mmproj), None)
        return {
            "model_id": mid,
            "display_name": self.repo_id,
            "engine": self.engine_hint if self.engine_hint != "unknown" else "llama.cpp",
            "format": self.format_hint,
            "repo": self.repo_id,
            "quant": self.recommended_quant or "auto",
            "profile": profile,
            "context": context,
            "weight_gb": self.weight_gb,
            "sampling": sampling or {},
            "variant": {
                "include": include,
                "mmproj": mmproj,
                "mtp": "mtp" in self.repo_id.lower(),
                "spark_moe_backend": (
                    "flashinfer_b12x" if self.format_hint == "nvfp4" else None
                ),
                "provider": self.repo_id.split("/")[0] if "/" in self.repo_id else "",
                "hub": True,
            },
            "estimate": {
                "weight_gb": self.weight_gb,
                "kv_gb": 0.0,
                "activation_gb": round(self.weight_gb * 0.12, 2),
                "os_reserve_gb": 8.0,
                "total_gb": round(self.weight_gb * 1.12 + 8.0, 2),
                "available_gb": 0.0,
                "fits": True,
                "headroom_gb": 0.0,
            },
            "reasons": list(self.notes),
            "rejected": False,
            "reject_reason": "",
            "score": 0.0,
        }


def _api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def infer_engine_and_format(repo_id: str, tags: list[str] | None = None) -> tuple[EngineHint, str]:
    tags_l = {t.lower() for t in (tags or [])}
    rid = repo_id.lower()
    if "nvfp4" in rid or "nvfp4" in tags_l:
        return "vllm", "nvfp4"
    if "mlx" in rid or "mlx" in tags_l or rid.startswith("mlx-community/"):
        return "mlx", "mlx"
    if "gguf" in rid or "gguf" in tags_l:
        return "llama.cpp", "gguf"
    if any(t in tags_l for t in ("safetensors", "text-generation", "image-text-to-text")):
        # Official dense/MoE checkpoints are best served with vLLM on Spark/CUDA
        return "vllm", "safetensors"
    return "unknown", "unknown"


def detect_quant(name: str) -> str | None:
    for pattern in QUANT_PATTERNS:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def infer_model_family(repo_id: str, families: Iterable[str]) -> str | None:
    """Match a Hub repo name to a catalog profile family."""
    normalized_repo = re.sub(r"[^a-z0-9]", "", repo_id.lower())
    matches = [
        family
        for family in families
        if re.sub(r"[^a-z0-9]", "", family.lower()) in normalized_repo
    ]
    return max(matches, key=len) if matches else None


def infer_active_params_b(repo_id: str) -> float | None:
    """Infer active parameter billions from common Hub repository names."""
    active = re.search(r"a(\d+(?:\.\d+)?)b(?!it)", repo_id, re.IGNORECASE)
    if active:
        return float(active.group(1))
    totals = re.findall(r"(\d+(?:\.\d+)?)b(?!it)", repo_id, re.IGNORECASE)
    return float(totals[-1]) if totals else None


def _org_bonus(repo_id: str) -> float:
    org = repo_id.split("/")[0] if "/" in repo_id else repo_id
    try:
        return float(20 - PREFERRED_ORGS.index(org))
    except ValueError:
        return 0.0


def search_hub(
    query: str,
    *,
    limit: int = 20,
    engine: EngineHint | None = None,
    curated_repos: set[str] | None = None,
) -> list[HubModelHit]:
    """Search Hugging Face models relevant to local llama.cpp / MLX / vLLM serving."""
    api = _api()
    curated = curated_repos or set()
    filters: list[str] = []
    if engine == "llama.cpp":
        filters.append("gguf")
    elif engine == "mlx":
        filters.append("mlx")
    # Broad search; we re-rank locally.
    models = list(
        api.list_models(
            search=query,
            filter=filters or None,
            apps="llama.cpp" if engine == "llama.cpp" else None,
            sort="downloads",
            limit=max(limit * 3, 30),
        )
    )
    hits: list[HubModelHit] = []
    for m in models:
        repo_id = m.id
        tags = list(getattr(m, "tags", None) or [])
        eng, fmt = infer_engine_and_format(repo_id, tags)
        if engine and eng != engine and eng != "unknown":
            continue
        if engine == "llama.cpp" and "gguf" not in repo_id.lower() and "gguf" not in {
            t.lower() for t in tags
        }:
            continue
        downloads = int(getattr(m, "downloads", 0) or 0)
        likes = int(getattr(m, "likes", 0) or 0)
        hit = HubModelHit(
            repo_id=repo_id,
            downloads=downloads,
            likes=likes,
            tags=tags,
            pipeline_tag=getattr(m, "pipeline_tag", None),
            engine_hint=eng,
            format_hint=fmt,
            curated=repo_id in curated,
        )
        score = _org_bonus(repo_id)
        score += min(downloads, 500_000) / 50_000.0
        score += min(likes, 5_000) / 500.0
        if hit.curated:
            score += 40
            hit.notes.append("in rondine curated catalog")
        if eng == "unknown":
            score -= 5
        # Prefer instruct / it / GGUF naming for coding
        rid_l = repo_id.lower()
        if any(x in rid_l for x in ("-it", "instruct", "gguf", "mlx", "nvfp4")):
            score += 3
        hit.score = score
        hits.append(hit)

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def inspect_repo(
    repo_id: str,
    *,
    prefer_quant: str | None = None,
) -> HubInspectResult:
    """Inspect a Hub repo and pick a default quant / weight size from file listing."""
    api = _api()
    info = api.model_info(repo_id, files_metadata=True)
    tags = list(getattr(info, "tags", None) or [])
    eng, fmt = infer_engine_and_format(repo_id, tags)
    files: list[HubFile] = []
    siblings = getattr(info, "siblings", None) or []
    for sib in siblings:
        path = getattr(sib, "rfilename", None) or getattr(sib, "path", None) or ""
        if not path:
            continue
        size = getattr(sib, "size", None) or 0
        size_gb = float(size) / (1024**3) if size else 0.0
        lower = path.lower()
        is_gguf = lower.endswith(".gguf")
        is_mmproj = "mmproj" in lower
        quant = detect_quant(path)
        # Skip tiny non-weight files
        if size_gb < 0.01 and not is_gguf and not lower.endswith(
            (".safetensors", ".npz", ".bin")
        ):
            continue
        if is_gguf or lower.endswith((".safetensors", ".npz", ".bin")) or "mlx" in lower:
            files.append(
                HubFile(
                    path=path,
                    size_gb=round(size_gb, 3),
                    quant=quant,
                    is_mmproj=is_mmproj,
                    is_gguf=is_gguf,
                )
            )

    notes: list[str] = []
    preferred_order = [
        prefer_quant,
        "UD-Q4_K_XL",
        "Q4_K_M",
        "Q4_K_L",
        "Q5_K_M",
        "Q4_K_S",
        "4bit",
        "NVFP4",
        "Q8_0",
    ]
    preferred_order = [q for q in preferred_order if q]

    weight_files = [f for f in files if not f.is_mmproj]
    recommended_quant: str | None = None
    recommended_file: str | None = None
    weight_gb = 0.0

    for quant in preferred_order:
        if not quant:
            continue
        matches = [
            f for f in weight_files if f.quant and f.quant.upper() == quant.upper()
        ]
        if not matches:
            matches = [f for f in weight_files if quant.lower() in f.path.lower()]
        if matches:
            family_sorted = sorted(
                matches,
                key=lambda f: (
                    0 if "00001-of-" in f.path or "-of-" not in f.path else 1,
                    f.path,
                ),
            )
            recommended_quant = family_sorted[0].quant or quant
            recommended_file = family_sorted[0].path
            token = recommended_quant or quant
            weight_gb = sum(
                f.size_gb
                for f in weight_files
                if token and (f.quant == token or token.lower() in f.path.lower())
            )
            if weight_gb <= 0:
                weight_gb = family_sorted[0].size_gb
            break

    if recommended_file is None and weight_files:
        # Largest single non-mmproj file as fallback
        biggest = max(weight_files, key=lambda f: f.size_gb)
        recommended_file = biggest.path
        recommended_quant = biggest.quant
        weight_gb = biggest.size_gb
        notes.append("no preferred quant matched; using largest weight file")

    if eng == "unknown" and any(f.is_gguf for f in files):
        eng, fmt = "llama.cpp", "gguf"
    if eng == "unknown" and "mlx" in repo_id.lower():
        eng, fmt = "mlx", "mlx"

    if weight_gb <= 0 and weight_files:
        weight_gb = sum(f.size_gb for f in weight_files)

    notes.append(f"engine hint: {eng} / format: {fmt}")
    if recommended_quant:
        notes.append(f"recommended quant: {recommended_quant} (~{weight_gb:.1f} GB)")

    return HubInspectResult(
        repo_id=repo_id,
        engine_hint=eng,
        format_hint=fmt,
        files=files,
        recommended_quant=recommended_quant,
        recommended_file=recommended_file,
        weight_gb=round(weight_gb, 2),
        tags=tags,
        downloads=int(getattr(info, "downloads", 0) or 0),
        notes=notes,
    )


def curated_repo_ids(catalog_models: list[Any]) -> set[str]:
    repos: set[str] = set()
    for model in catalog_models:
        for variant in getattr(model, "variants", []):
            repos.add(variant.repo)
    return repos


def apply_hub_hardware_budget(
    selected: dict[str, Any],
    catalog: Catalog,
    hw: HardwareInfo,
    *,
    memory_mode: str = "auto",
    allow_oversize: bool = False,
) -> tuple[float, str]:
    """Recalculate a Hub plan against this host's usable memory."""
    if memory_mode == "mmap" and not allow_oversize:
        raise ValueError("--memory-mode mmap requires --allow-oversize")
    available_gb, reserve_gb = available_memory_gb(catalog, hw)
    estimate = selected["estimate"]
    weight_gb = float(estimate["weight_gb"])
    activation_gb = float(estimate["activation_gb"])
    kv_gb = float(estimate.get("kv_gb") or 0.0)
    active_params_b = infer_active_params_b(str(selected.get("repo") or ""))
    context = int(selected.get("context") or 0)
    if kv_gb <= 0 and active_params_b and context > 0:
        kv_gb = round(
            context * active_params_b * catalog.policy.kv_bytes_per_token_per_b,
            2,
        )
        estimate["kv_gb"] = kv_gb
        selected.setdefault("reasons", []).append(
            f"estimated KV cache from ~{active_params_b:g}B active parameters"
        )
    total_gb = round(weight_gb + activation_gb + kv_gb + reserve_gb, 2)
    resident_fits = available_gb >= total_gb
    mode = "resident"
    if (
        memory_mode == "hybrid"
        or (
            memory_mode == "auto"
            and not resident_fits
            and hw.is_discrete_cuda
            and selected.get("engine") == "llama.cpp"
            and selected.get("format") == "gguf"
        )
    ):
        if (
            not hw.is_discrete_cuda
            or selected.get("engine") != "llama.cpp"
            or selected.get("format") != "gguf"
        ):
            raise ValueError(
                "hybrid mode requires llama.cpp GGUF on a discrete CUDA host"
            )
        mode = "hybrid"
        available_gb = hw.ram_gb + hw.vram_gb
        reserve_gb = catalog.policy.os_reserve_gb + catalog.policy.vram_reserve_gb
        total_gb = round(weight_gb + activation_gb + kv_gb + reserve_gb, 2)
        selected["engine_args"] = {
            **dict(selected.get("engine_args") or {}),
            "n_gpu_layers": "auto",
            "fit": True,
            "fit_target": 1536,
            "mmap": True,
            "mlock": False,
        }
    elif memory_mode == "mmap":
        if selected.get("engine") != "llama.cpp" or selected.get("format") != "gguf":
            raise ValueError("mmap mode requires a llama.cpp GGUF repo")
        mode = "mmap"
        selected["experimental"] = True
        selected["warnings"] = [
            "experimental SSD demand paging; likely unusably slow and not a supported fit"
        ]
        selected["engine_args"] = {
            **dict(selected.get("engine_args") or {}),
            "n_gpu_layers": 0 if hw.is_apple_silicon else "auto",
            "fit": not hw.is_apple_silicon,
            "fit_target": 2048,
            "mmap": True,
            "mlock": False,
            "parallel": 1,
            "batch_size": 128,
            "ubatch_size": 64,
            "cache_type_k": "q4_1",
            "cache_type_v": "q4_1",
        }
    disk_required = round(weight_gb * 1.05, 2)
    estimate.update(
        {
            "os_reserve_gb": round(reserve_gb, 2),
            "total_gb": total_gb,
            "available_gb": available_gb,
            "headroom_gb": round(available_gb - total_gb, 2),
            "fits": available_gb >= total_gb,
            "memory_mode": mode,
            "experimental": mode == "mmap",
            "resident_shortfall_gb": round(max(0.0, total_gb - available_gb), 2),
            "disk_required_gb": disk_required,
            "disk_available_gb": hw.disk_free_gb,
        }
    )
    selected["memory_mode"] = mode
    if mode == "mmap" and hw.disk_free_gb < disk_required:
        raise ValueError(
            f"insufficient disk: need ~{disk_required:.0f}GB free, "
            f"have {hw.disk_free_gb:.0f}GB"
        )
    unit = "VRAM" if hw.is_discrete_cuda else "RAM"
    if mode == "hybrid":
        unit = "combined RAM+VRAM"
    return total_gb, unit
