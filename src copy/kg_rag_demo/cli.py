from __future__ import annotations

import argparse
import json

from .pipeline import DemoPipeline


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

    debug_parser = subparsers.add_parser("debug", help="Show retrieved context")
    debug_parser.add_argument("question", help="Question in Chinese")

    args = parser.parse_args()
    pipeline = DemoPipeline(
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
