from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Search processed chunks by doc id and keywords.")
    parser.add_argument("terms", nargs="+", help="Keywords; a chunk matches if any term is present.")
    parser.add_argument("--doc-id", default="", help="Optional exact doc_id filter.")
    parser.add_argument("--chunks", default="processed_data/chunks.jsonl")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=1400)
    args = parser.parse_args()

    path = Path(args.chunks)
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if args.doc_id and chunk.get("doc_id") != args.doc_id:
                continue
            text = chunk.get("text", "")
            if not any(term in text for term in args.terms):
                continue
            count += 1
            snippet = " ".join(text.split())
            print(
                f"[{count}] line={line_no} doc={chunk.get('doc_id')} "
                f"chunk={chunk.get('chunk_id')} page={chunk.get('page')}"
            )
            print(snippet[: args.max_chars])
            print("---")
            if count >= args.limit:
                break

    if count == 0:
        print("No chunks matched.")


if __name__ == "__main__":
    main()
