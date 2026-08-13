from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.solver import ordered_values_by_questions, read_question_cache, write_answer_rows, write_evidence_jsonl


def load_questions(root: Path) -> list[dict]:
    qdir = root / "public_dataset_a" / "public_dataset_upload" / "questions" / "group_a"
    questions: list[dict] = []
    for path in sorted(qdir.glob("*.json")):
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    return questions


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild checkpoint CSV/JSONL from atomic per-question cache.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--cache-dir", default="run_cache/questions")
    parser.add_argument("--checkpoint-csv", default="answer.checkpoint.csv")
    parser.add_argument("--checkpoint-evidence", default="evidence.checkpoint.jsonl")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cache_dir = resolve_path(root, args.cache_dir)
    checkpoint_csv = resolve_path(root, args.checkpoint_csv)
    checkpoint_evidence = resolve_path(root, args.checkpoint_evidence)
    questions = load_questions(root)
    qids = {q["qid"] for q in questions}

    cache_rows, cache_evidence_rows, bad_cache_files = read_question_cache(cache_dir)
    row_by_qid = {qid: row for qid, row in cache_rows.items() if qid in qids}
    evidence_by_qid = {qid: row for qid, row in cache_evidence_rows.items() if qid in qids}

    ordered_rows = ordered_values_by_questions(row_by_qid, questions)
    ordered_evidence = ordered_values_by_questions(evidence_by_qid, questions)
    write_answer_rows(checkpoint_csv, ordered_rows)
    write_evidence_jsonl(checkpoint_evidence, ordered_evidence)

    print(f"cache_dir: {cache_dir}")
    print(f"rebuilt checkpoint_csv: {checkpoint_csv}")
    print(f"rebuilt checkpoint_evidence: {checkpoint_evidence}")
    print(f"answer_rows: {len(ordered_rows)}/{len(questions)}")
    print(f"evidence_rows: {len(ordered_evidence)}/{len(questions)}")
    if bad_cache_files:
        print(f"bad_cache_files: {len(bad_cache_files)}")


if __name__ == "__main__":
    main()
