from __future__ import annotations

import shutil
from typing import Any

import chromadb

from .config import CHROMA_DIR, Settings
from .llm import LLMClient
from .models import ChunkRecord


class VectorStore:
    def __init__(self, settings: Settings, llm: LLMClient, reset: bool = False) -> None:
        self.settings = settings
        self.llm = llm
        if reset and CHROMA_DIR.exists():
            shutil.rmtree(CHROMA_DIR)
        self.client: chromadb.ClientAPI | None = None
        self.collection: Any | None = None
        self._open_client()

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> int:
        if not chunks:
            return 0

        collection = self._require_collection()
        texts = [chunk.text for chunk in chunks]
        embeddings = self.llm.embed_documents(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._to_metadata(chunk) for chunk in chunks]
        documents = texts

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        return len(chunks)

    def query(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or self.settings.retrieval_k
        query_embedding = self.llm.embed_query(question)
        result = self._query_collection(
            collection=self._require_collection(),
            query_embedding=query_embedding,
            top_k=k,
        )
        hits: list[dict[str, Any]] = []

        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            item = dict(meta or {})
            item["chunk_id"] = chunk_id
            item["text"] = text
            item["distance"] = distance
            hits.append(item)
        return hits

    def self_check(self, query_text: str) -> dict[str, Any]:
        query_embedding = self.llm.embed_query(query_text)
        try:
            self.close()
            return self._reopen_and_query(query_embedding)
        finally:
            self._open_client()

    def close(self) -> None:
        if self.client is None:
            return
        self.client.close()
        self.client = None
        self.collection = None

    def _open_client(self) -> None:
        if self.client is not None:
            return
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(name=self.settings.collection_name)

    def _require_collection(self) -> Any:
        if self.collection is None:
            self._open_client()
        if self.collection is None:
            raise RuntimeError("Chroma collection 未成功初始化。")
        return self.collection

    @staticmethod
    def _query_collection(collection: Any, query_embedding: list[float], top_k: int) -> dict[str, Any]:
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def _reopen_and_query(self, query_embedding: list[float]) -> dict[str, Any]:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            collection = client.get_collection(name=self.settings.collection_name)
            count = collection.count()
            if count == 0:
                raise RuntimeError("collection 为空，没有可查询的数据。")

            result = self._query_collection(
                collection=collection,
                query_embedding=query_embedding,
                top_k=1,
            )
            ids = result.get("ids", [[]])[0]
            if not ids:
                raise RuntimeError("能够重新打开 collection，但检索结果为空。")

            return {
                "count": count,
                "top_hit_id": ids[0],
            }
        except Exception as exc:
            raise RuntimeError(f"向量库重开自检失败：{exc}") from exc
        finally:
            client.close()

    @staticmethod
    def _to_metadata(chunk: ChunkRecord) -> dict[str, Any]:
        metadata = {
            "doc_id": chunk.doc_id,
            "source_path": chunk.source_path,
            "title": chunk.title,
            "modality": chunk.modality,
            "page_number": chunk.page_number or -1,
            "order": chunk.order,
        }
        metadata.update(chunk.metadata)
        return metadata
