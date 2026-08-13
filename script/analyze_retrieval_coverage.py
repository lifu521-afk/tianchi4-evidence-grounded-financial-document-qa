from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.solver import format_evidence, load_processed
from agent.retrieval import LexicalIndex, gather_evidence


LETTERS = "ABCD"


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval coverage without calling the model.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--output", default="retrieval_coverage_report.csv")
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    questions, chunks = load_processed(root / "processed_data")
    if args.limit is not None:
        questions = questions[: args.limit]
    index = LexicalIndex(chunks)
    output = resolve_path(root, args.output)

    rows: list[dict[str, str | int]] = []
    domain_counts: Counter[str] = Counter()
    domain_weak: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()

    for question in questions:
        evidence = gather_evidence(index, question, max_chars=args.max_context_chars or None)
        doc_ids = [str(doc_id) for doc_id in question.get("doc_ids") or []]
        hit_docs = {str(item.get("doc_id")) for item in evidence if item.get("doc_id")}
        option_letters = [str(key).upper() for key in sorted(question.get("options", {})) if str(key).upper() in LETTERS]
        option_letters = option_letters or list(LETTERS)
        option_hits = {
            letter: sum(1 for item in evidence if f"option_{letter}" in item.get("sources", []))
            for letter in option_letters
        }
        missing_docs = [doc_id for doc_id in doc_ids if doc_id not in hit_docs]
        missing_options = [letter for letter, count in option_hits.items() if count == 0]
        prompt_chars = len(format_evidence(evidence, question, "compact"))
        flags: list[str] = []
        if missing_docs:
            flags.append("missing_doc")
        if missing_options:
            flags.append("missing_option")
        if len(evidence) < 6:
            flags.append("low_evidence")
        if prompt_chars > 18000:
            flags.append("large_prompt")
        domain = str(question.get("domain", ""))
        domain_counts[domain] += 1
        if flags:
            domain_weak[domain] += 1
        flag_counts.update(flags)
        rows.append(
            {
                "qid": question["qid"],
                "domain": domain,
                "answer_format": question.get("answer_format", ""),
                "docs_expected": len(doc_ids),
                "docs_hit": len(hit_docs & set(doc_ids)) if doc_ids else len(hit_docs),
                "missing_docs": ";".join(missing_docs),
                "evidence_count": len(evidence),
                "prompt_chars_compact": prompt_chars,
                "option_A_hits": option_hits.get("A", 0),
                "option_B_hits": option_hits.get("B", 0),
                "option_C_hits": option_hits.get("C", 0),
                "option_D_hits": option_hits.get("D", 0),
                "missing_options": "".join(missing_options),
                "flags": ";".join(flags),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    print(f"output: {output}")
    print(f"questions: {len(rows)}")
    print(f"flags: {dict(flag_counts)}")
    for domain, total in sorted(domain_counts.items()):
        print(f"domain {domain}: weak={domain_weak[domain]}/{total}")
    weak_rows = [row for row in rows if row["flags"]]
    if weak_rows:
        print("weak_first_10:")
        for row in weak_rows[:10]:
            print(f"  {row['qid']} {row['flags']} missing_options={row['missing_options']} missing_docs={row['missing_docs']}")


if __name__ == "__main__":
    main()
