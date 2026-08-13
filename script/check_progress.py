from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.solver import read_question_cache


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
    parser = argparse.ArgumentParser(description="Show answer checkpoint/cache progress.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--checkpoint-csv", default="answer.checkpoint.csv")
    parser.add_argument("--cache-dir", default="run_cache/questions")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    checkpoint = resolve_path(root, args.checkpoint_csv)
    cache_dir = resolve_path(root, args.cache_dir) if args.cache_dir else None

    questions = load_questions(root)
    qids = [q["qid"] for q in questions]

    checkpoint_completed: list[str] = []
    if checkpoint.exists():
        with checkpoint.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("qid") and row["qid"] != "summary":
                    checkpoint_completed.append(row["qid"])

    cache_rows, _, bad_cache_files = read_question_cache(cache_dir)
    checkpoint_set = set(checkpoint_completed)
    cache_set = set(cache_rows)
    completed_set = (checkpoint_set | cache_set) & set(qids)
    missing = [qid for qid in qids if qid not in completed_set]

    print(f"checkpoint: {checkpoint}")
    print(f"cache_dir: {cache_dir}")
    print(f"checkpoint_completed: {len(checkpoint_set & set(qids))}/{len(qids)}")
    print(f"cache_completed: {len(cache_set & set(qids))}/{len(qids)}")
    print(f"combined_completed: {len(completed_set)}/{len(qids)}")
    if bad_cache_files:
        print(f"bad_cache_files: {len(bad_cache_files)}")
    if missing:
        print(f"next: {missing[0]}")
        print(f"missing_first_10: {missing[:10]}")
    else:
        print("all questions completed")


if __name__ == "__main__":
    main()
