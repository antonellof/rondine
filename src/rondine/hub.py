"""Hugging Face Hub discovery — complements the curated catalog."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

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
