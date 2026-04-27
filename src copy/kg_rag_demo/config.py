from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# utf-8-sig: avoid a BOM on the first line breaking OPENAI_API_KEY parsing on Windows editors.
# override=True: if the shell or OS already has OPENAI_API_KEY, still apply values from .env
# (common when another machine has a stale or wrong key in user/system environment).
load_dotenv(encoding="utf-8-sig", override=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_local_embed_model(spec: str) -> str:
    """Resolve LOCAL_EMBED_MODEL to a filesystem path when a project-local HF cache exists."""
    raw = (spec or "").strip()
    if not raw:
        return raw

    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.is_dir():
        return str(candidate.resolve())

    rooted = PROJECT_ROOT / raw
    if rooted.is_dir():
        return str(rooted.resolve())

    if "/" not in raw or "\\" in raw or raw.startswith("."):
        return raw

    parts = raw.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return raw

    cache_root = PROJECT_ROOT / "model" / f"models--{parts[0]}--{parts[1]}"
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.is_dir():
        return raw

    ref_main = cache_root / "refs" / "main"
    if ref_main.is_file():
        commit = ref_main.read_text(encoding="utf-8").strip()
        snap = snapshots_dir / commit
        if snap.is_dir():
            return str(snap.resolve())

    for child in sorted(snapshots_dir.iterdir()):
        if child.is_dir():
            return str(child.resolve())

    return raw
DATA_DIR = PROJECT_ROOT / "data"
WORK_DIR = PROJECT_ROOT / "workdir"
CHROMA_DIR = WORK_DIR / "chroma"
CACHE_DIR = WORK_DIR / "cache"


@dataclass
class Settings:
    openai_api_key: str = (os.getenv("OPENAI_API_KEY", "") or "").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_timeout_seconds: float = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
    openai_max_retries: int = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "2")))
    openai_retry_backoff_seconds: float = float(os.getenv("OPENAI_RETRY_BACKOFF_SECONDS", "2"))
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local")
    local_embed_model: str = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "kg_rag_demo")
    graph_extract_workers: int = max(1, int(os.getenv("GRAPH_EXTRACT_WORKERS", "4")))
    graph_progress_heartbeat_seconds: float = float(os.getenv("GRAPH_PROGRESS_HEARTBEAT_SECONDS", "5"))


def ensure_project_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
