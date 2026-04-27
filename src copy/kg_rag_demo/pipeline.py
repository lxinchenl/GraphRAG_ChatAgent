from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from typing import Callable

from .chunking import chunk_documents
from .config import DATA_DIR, Settings, ensure_project_dirs
from .graph_store import GraphStore
from .llm import LLMClient
from .models import ChunkRecord, QueryAnswer, RelationEdge
from .parsers import DocumentParser
from .vector_store import VectorStore


class DemoPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        progress_callback: Callable[[str], None] | None = None,
        *,
        enable_graph: bool = True,
        reset_vector_store: bool = False,
    ) -> None:
        self.settings = settings or Settings()
        self.progress_callback = progress_callback or print
        self.enable_graph = enable_graph
        ensure_project_dirs()
        self.parser = DocumentParser(progress_callback=self._log)
        self.llm = LLMClient(self.settings)
        self.vector_store = VectorStore(self.settings, self.llm, reset=reset_vector_store)
        self.graph_store: GraphStore | None = None
        if enable_graph:
            self.graph_store = GraphStore(
                uri=self.settings.neo4j_uri,
                username=self.settings.neo4j_username,
                password=self.settings.neo4j_password,
            )

    def close(self) -> None:
        self.vector_store.close()
        if self.graph_store is not None:
            self.graph_store.close()

    def ingest(self, data_dir: str | None = None) -> dict[str, int]:
        graph_store = self._require_graph_store()
        docs, chunks, inserted_chunks = self._ingest_vectors(data_dir=data_dir, prefix="ingest")

        self._log("[ingest] 初始化 Neo4j 约束")
        graph_store.ensure_constraints()

        relation_count = 0
        total_chunks = len(chunks)
        graph_extract_workers = max(1, self.settings.graph_extract_workers)
        if graph_extract_workers == 1:
            for index, chunk in enumerate(chunks, start=1):
                self._log(
                    f"[graph] 处理 chunk {index}/{total_chunks}: "
                    f"{chunk.title} | 页码 {chunk.page_number if chunk.page_number is not None else '-'} | 长度 {len(chunk.text)}"
                )
                graph_store.upsert_chunk(chunk)
                self._log(f"[graph] 开始抽取关系: {chunk.chunk_id}")
                _, relations = self.llm.extract_graph(chunk.text)
                self._log(f"[graph] 抽取完成，得到 {len(relations)} 条关系")
                relation_count += graph_store.upsert_relations(chunk, relations)
                self._log(f"[graph] 已累计写入 {relation_count} 条关系")
        else:
            self._log(f"[graph] 启用并发关系抽取，并发数: {graph_extract_workers}")
            future_to_chunk: dict[Future[list[RelationEdge]], tuple[int, ChunkRecord]] = {}
            completed_chunks = 0
            next_chunk_index = 1
            heartbeat_seconds = max(1.0, self.settings.graph_progress_heartbeat_seconds)
            last_heartbeat_at = time.monotonic()

            def submit_chunk(chunk_index: int, chunk_record: ChunkRecord, executor: ThreadPoolExecutor) -> None:
                nonlocal next_chunk_index
                self._log(
                    f"[graph] 处理 chunk {chunk_index}/{total_chunks}: "
                    f"{chunk_record.title} | 页码 {chunk_record.page_number if chunk_record.page_number is not None else '-'} | 长度 {len(chunk_record.text)}"
                )
                graph_store.upsert_chunk(chunk_record)
                self._log(f"[graph] 提交抽取任务: {chunk_record.chunk_id}")
                future = executor.submit(self._extract_graph_relations, chunk_record.text)
                future_to_chunk[future] = (chunk_index, chunk_record)
                next_chunk_index += 1

            with ThreadPoolExecutor(max_workers=graph_extract_workers, thread_name_prefix="graph-extract") as executor:
                while next_chunk_index <= total_chunks and len(future_to_chunk) < graph_extract_workers:
                    submit_chunk(next_chunk_index, chunks[next_chunk_index - 1], executor)

                while future_to_chunk:
                    done, _ = wait(
                        list(future_to_chunk.keys()),
                        timeout=heartbeat_seconds,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        pending_submit = total_chunks - next_chunk_index + 1
                        self._log(
                            f"[graph] 抽取进行中，已完成 {completed_chunks}/{total_chunks}，"
                            f"进行中 {len(future_to_chunk)}，待提交 {max(0, pending_submit)}"
                        )
                        last_heartbeat_at = time.monotonic()
                        continue

                    for future in done:
                        index, chunk = future_to_chunk.pop(future)
                        try:
                            relations = future.result()
                        except Exception as exc:
                            completed_chunks += 1
                            self._log(f"[graph] 抽取失败 {index}/{total_chunks}: {chunk.chunk_id} | {exc}")
                        else:
                            completed_chunks += 1
                            self._log(f"[graph] 抽取完成 {index}/{total_chunks}，得到 {len(relations)} 条关系: {chunk.chunk_id}")
                            relation_count += graph_store.upsert_relations(chunk, relations)
                            self._log(f"[graph] 已累计写入 {relation_count} 条关系")

                        while next_chunk_index <= total_chunks and len(future_to_chunk) < graph_extract_workers:
                            submit_chunk(next_chunk_index, chunks[next_chunk_index - 1], executor)

                    now = time.monotonic()
                    if now - last_heartbeat_at >= heartbeat_seconds:
                        pending_submit = total_chunks - next_chunk_index + 1
                        self._log(
                            f"[graph] 抽取进行中，已完成 {completed_chunks}/{total_chunks}，"
                            f"进行中 {len(future_to_chunk)}，待提交 {max(0, pending_submit)}"
                        )
                        last_heartbeat_at = now

        self._log("[ingest] 全部流程完成")
        return {
            "documents": len(docs),
            "chunks": inserted_chunks,
            "relations": relation_count,
        }

    def ingest_vectors(self, data_dir: str | None = None) -> dict[str, object]:
        docs, chunks, inserted_chunks = self._ingest_vectors(data_dir=data_dir, prefix="vector")
        self_check = self._self_check_vector_store(chunks)
        self._log("[vector] 全部流程完成")
        return {
            "documents": len(docs),
            "chunks": inserted_chunks,
            "vector_self_check": self_check["status"],
        }

    def ask(self, question: str) -> QueryAnswer:
        self._log(f"[ask] 收到问题: {question}")
        chunk_hits = self.vector_store.query(question, top_k=self.settings.retrieval_k)
        self._log(f"[ask] 向量检索完成，命中 {len(chunk_hits)} 个片段")
        query_entities = self.llm.extract_query_entities(question)
        self._log(f"[ask] 问题实体抽取结果: {query_entities or '无'}")
        graph_hits: list[dict[str, object]] = []
        if self.graph_store is not None:
            graph_hits = self.graph_store.query_entity_relations(query_entities, limit=self.settings.retrieval_k * 2)
        self._log(f"[ask] 图谱检索完成，命中 {len(graph_hits)} 条关系")
        answer = self.llm.answer_question(question, chunk_hits, graph_hits)
        return QueryAnswer(
            answer=answer,
            chunk_hits=chunk_hits,
            graph_hits=graph_hits,
            query_entities=query_entities,
        )

    def debug_context(self, question: str) -> dict[str, object]:
        answer = self.ask(question)
        return asdict(answer)

    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)

    def _extract_graph_relations(self, text: str) -> list[RelationEdge]:
        worker_llm = LLMClient(self.settings)
        _, relations = worker_llm.extract_graph(text)
        return relations

    def _ingest_vectors(self, data_dir: str | None, prefix: str) -> tuple[list[object], list[ChunkRecord], int]:
        target_dir = data_dir or str(DATA_DIR)
        self._log(f"[{prefix}] 开始构建向量索引，数据目录: {target_dir}")
        docs = self.parser.parse_directory(target_dir)
        self._log(f"[{prefix}] 文档解析完成，共得到 {len(docs)} 条记录，开始切块")
        chunks = chunk_documents(docs, self.settings)
        self._log(f"[{prefix}] 切块完成，共生成 {len(chunks)} 个 chunk")
        self._log(f"[{prefix}] 开始写入 Chroma 向量库")
        inserted_chunks = self.vector_store.upsert_chunks(chunks)
        self._log(f"[{prefix}] 向量写入完成，共写入 {inserted_chunks} 个 chunk")
        return docs, chunks, inserted_chunks

    def _self_check_vector_store(self, chunks: list[ChunkRecord]) -> dict[str, str]:
        if not chunks:
            self._log("[vector] 跳过自检：没有生成任何 chunk")
            return {"status": "skipped"}

        probe_text = chunks[0].text.strip() or chunks[0].title.strip() or "测试"
        self._log("[vector] 开始自检：重新打开 Chroma 并执行一次检索")
        check_result = self.vector_store.self_check(probe_text)
        self._log(
            f"[vector] 自检通过：collection={check_result['count']}，"
            f"top_hit={check_result['top_hit_id']}"
        )
        return {"status": "passed"}

    def _require_graph_store(self) -> GraphStore:
        if self.graph_store is None:
            raise RuntimeError("当前 pipeline 未启用 Neo4j，无法构建知识图谱。")
        return self.graph_store
