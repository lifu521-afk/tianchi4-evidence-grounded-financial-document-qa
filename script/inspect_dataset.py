from __future__ import annotations

import collections
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import default_paths
from agent.preprocess import discover_raw_documents, load_questions


def main() -> None:
    paths = default_paths(ROOT)
    questions = load_questions(paths.questions_dir)
    raw_docs = discover_raw_documents(paths.raw_dir)
    print(f"questions: {len(questions)}")
    print("by_domain:", dict(collections.Counter(q.domain for q in questions)))
    print("by_answer_format:", dict(collections.Counter(q.answer_format for q in questions)))
    print(f"raw_documents: {len(raw_docs)}")
    missing = sorted({doc_id for q in questions for doc_id in q.doc_ids if doc_id not in raw_docs})
    print(f"missing_doc_ids: {len(missing)}")
    for doc_id in missing:
        print("  ", doc_id)


if __name__ == "__main__":
    main()

