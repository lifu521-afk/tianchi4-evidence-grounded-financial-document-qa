from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import default_paths
from agent.solver import load_processed
from agent.retrieval import LexicalIndex, gather_evidence, build_query, expand_query_terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect retrieved evidence for one question.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--qid", required=True, help="Question id, for example fin_a_001.")
    parser.add_argument("--max-chars", type=int, default=900, help="Max characters to print per evidence chunk.")
    args = parser.parse_args()

    paths = default_paths(args.root)
    questions, chunks = load_processed(paths.processed_dir)
    question = next((q for q in questions if q["qid"] == args.qid), None)
    if question is None:
        raise SystemExit(f"qid not found: {args.qid}")

    index = LexicalIndex(chunks)
    evidence = gather_evidence(index, question)

    print(f"QID: {question['qid']}")
    print(f"DOMAIN: {question.get('domain')} FORMAT: {question.get('answer_format')}")
    print(f"DOCS: {', '.join(question.get('doc_ids', []))}")
    print("QUESTION:")
    print(question["question"])
    print("OPTIONS:")
    for key, value in sorted(question.get("options", {}).items()):
        print(f"  {key}. {value}")
    print("EXPANDED TERMS:")
    terms = expand_query_terms(build_query(question))
    print("  " + ", ".join(terms))
    print("EVIDENCE:")
    for idx, item in enumerate(evidence, start=1):
        text = item.get("text", "").replace("\n", " ")
        if len(text) > args.max_chars:
            text = text[: args.max_chars] + "..."
        sources = ",".join(item.get("sources", []))
        print(f"[{idx}] doc={item['doc_id']} chunk={item['chunk_id']} page={item.get('page')} score={item.get('score')} sources={sources}")
        print(text)
        print()


if __name__ == "__main__":
    main()
