from __future__ import annotations

import argparse
import json

from .pipeline import AskRuntimeConfig, DemoPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge graph + RAG demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse files and build indexes")
    ingest_parser.add_argument("--data-dir", default=None, help="Directory that stores documents")

    ingest_vector_parser = subparsers.add_parser("ingest-vector", help="Only build the Chroma vector index")
    ingest_vector_parser.add_argument("--data-dir", default=None, help="Directory that stores documents")
    ingest_vector_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Chroma data before rebuilding the vector index",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", help="Question in Chinese")
    ask_parser.add_argument("--no-query-rewrite", action="store_true", help="Disable multi-query rewriting")
    ask_parser.add_argument("--no-hybrid-retrieval", action="store_true", help="Disable BM25 + dense hybrid retrieval")
    ask_parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    ask_parser.add_argument("--no-evidence-compression", action="store_true", help="Disable evidence compression")
    ask_parser.add_argument("--no-graph-entity-synonyms", action="store_true", help="Disable graph entity synonym expansion")
    ask_parser.add_argument("--no-graph-multi-hop", action="store_true", help="Disable graph multi-hop retrieval")
    ask_parser.add_argument("--no-graph-hit-dedup", action="store_true", help="Disable graph hit deduplication")
    ask_parser.add_argument("--no-graph-hit-rerank", action="store_true", help="Disable graph hit scoring/ranking")
    ask_parser.add_argument("--no-graph-hit-truncate", action="store_true", help="Disable graph hit truncation")
    ask_parser.add_argument("--graph-max-hops", type=int, default=None, help="Max hops for graph retrieval")
    ask_parser.add_argument("--graph-top-k", type=int, default=None, help="Final graph relation count after postprocess")
    ask_parser.add_argument(
        "--graph-synonyms-per-entity",
        type=int,
        default=None,
        help="Max synonym expansions per extracted entity",
    )
    ask_parser.add_argument("--debug-mode", action="store_true", help="Print debug traces for retrieval")
    ask_parser.add_argument("--retrieval-k", type=int, default=None, help="Final number of chunks for answering")
    ask_parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidates per rewritten query before dedupe/rerank",
    )

    debug_parser = subparsers.add_parser("debug", help="Show retrieved context")
    debug_parser.add_argument("question", help="Question in Chinese")
    debug_parser.add_argument("--no-query-rewrite", action="store_true", help="Disable multi-query rewriting")
    debug_parser.add_argument("--no-hybrid-retrieval", action="store_true", help="Disable BM25 + dense hybrid retrieval")
    debug_parser.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    debug_parser.add_argument("--no-evidence-compression", action="store_true", help="Disable evidence compression")
    debug_parser.add_argument("--no-graph-entity-synonyms", action="store_true", help="Disable graph entity synonym expansion")
    debug_parser.add_argument("--no-graph-multi-hop", action="store_true", help="Disable graph multi-hop retrieval")
    debug_parser.add_argument("--no-graph-hit-dedup", action="store_true", help="Disable graph hit deduplication")
    debug_parser.add_argument("--no-graph-hit-rerank", action="store_true", help="Disable graph hit scoring/ranking")
    debug_parser.add_argument("--no-graph-hit-truncate", action="store_true", help="Disable graph hit truncation")
    debug_parser.add_argument("--graph-max-hops", type=int, default=None, help="Max hops for graph retrieval")
    debug_parser.add_argument("--graph-top-k", type=int, default=None, help="Final graph relation count after postprocess")
    debug_parser.add_argument(
        "--graph-synonyms-per-entity",
        type=int,
        default=None,
        help="Max synonym expansions per extracted entity",
    )
    debug_parser.add_argument("--debug-mode", action="store_true", default=True, help="Print retrieval debug traces")
    debug_parser.add_argument("--retrieval-k", type=int, default=None, help="Final number of chunks for answering")
    debug_parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help="Candidates per rewritten query before dedupe/rerank",
    )

    args = parser.parse_args()
    ask_config = AskRuntimeConfig(
        enable_query_rewrite=not bool(getattr(args, "no_query_rewrite", False)),
        enable_hybrid_retrieval=not bool(getattr(args, "no_hybrid_retrieval", False)),
        enable_rerank=not bool(getattr(args, "no_rerank", False)),
        enable_evidence_compression=not bool(getattr(args, "no_evidence_compression", False)),
        enable_graph_entity_synonyms=not bool(getattr(args, "no_graph_entity_synonyms", False)),
        enable_graph_multi_hop=not bool(getattr(args, "no_graph_multi_hop", False)),
        enable_graph_hit_dedup=not bool(getattr(args, "no_graph_hit_dedup", False)),
        enable_graph_hit_rerank=not bool(getattr(args, "no_graph_hit_rerank", False)),
        enable_graph_hit_truncate=not bool(getattr(args, "no_graph_hit_truncate", False)),
        debug_mode=bool(getattr(args, "debug_mode", False)),
        retrieval_k=max(1, int(getattr(args, "retrieval_k", 5) or 5)),
        retrieval_candidate_k=max(
            1,
            int(getattr(args, "candidate_k", 12) or 12),
        ),
        graph_max_hops=max(1, int(getattr(args, "graph_max_hops", 2) or 2)),
        graph_synonyms_per_entity=max(0, int(getattr(args, "graph_synonyms_per_entity", 2) or 2)),
        graph_top_k=max(1, int(getattr(args, "graph_top_k", 8) or 8)),
    )
    ask_config.retrieval_candidate_k = max(ask_config.retrieval_k, ask_config.retrieval_candidate_k)

    pipeline = DemoPipeline(
        ask_config=ask_config,
        enable_graph=args.command != "ingest-vector",
        reset_vector_store=bool(getattr(args, "reset", False)),
    )
    try:
        if args.command == "ingest":
            result = pipeline.ingest(data_dir=args.data_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "ingest-vector":
            result = pipeline.ingest_vectors(data_dir=args.data_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "ask":
            result = pipeline.ask(args.question)
            print(result.answer)
        elif args.command == "debug":
            result = pipeline.debug_context(args.question)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
