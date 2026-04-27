from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedDocument:
    source_path: str
    doc_id: str
    title: str
    text: str
    modality: str
    page_number: int | None = None
    extra_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    text: str
    modality: str
    page_number: int | None
    order: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode:
    name: str
    label: str = "Entity"


@dataclass
class RelationEdge:
    source: str
    target: str
    relation: str
    evidence: str


@dataclass
class QueryAnswer:
    answer: str
    chunk_hits: list[dict[str, Any]]
    graph_hits: list[dict[str, Any]]
    query_entities: list[str] = field(default_factory=list)
