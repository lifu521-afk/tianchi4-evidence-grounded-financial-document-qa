from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.solver import extract_json_object, normalize_answer, review_risk_reasons


def load_questions(root: Path) -> dict[str, dict]:
    questions: list[dict] = []
    qdir = root / "public_dataset_a" / "public_dataset_upload" / "questions" / "group_a"
    for path in sorted(qdir.glob("*.json")):
        questions.extend(json.loads(path.read_text(encoding="utf-8")))
    return {q["qid"]: q for q in questions}


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated answers and review-risk profile without API calls.")
    parser.add_argument("--root", default=str(ROOT), help="Project root directory.")
    parser.add_argument("--answer-csv", default="answer.csv")
    parser.add_argument("--evidence-json", default="evidence.json")
    parser.add_argument("--suspect-output", default="answer_suspects.csv")
    parser.add_argument(
        "--risk-scope",
        choices=["targeted", "broad"],
        default="targeted",
        help="targeted lists high-risk reruns; broad also flags every multi-choice question.",
    )
    parser.add_argument(
        "--analysis-stage",
        choices=["final", "initial"],
        default="final",
        help="final evaluates reviewed output when available; initial reproduces first-pass risk analysis.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    answer_csv = resolve_path(root, args.answer_csv)
    evidence_json = resolve_path(root, args.evidence_json)
    suspect_output = resolve_path(root, args.suspect_output)
    qmap = load_questions(root)

    with answer_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("qid") != "summary"]
    evidence_rows = json.loads(evidence_json.read_text(encoding="utf-8"))
    emap = {row["qid"]: row for row in evidence_rows}

    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    by_format: dict[str, Counter[str]] = defaultdict(Counter)
    reviewed = Counter()
    reason_counts = Counter()
    suspects: list[dict[str, str]] = []

    for row in rows:
        qid = row["qid"]
        question = qmap[qid]
        evidence = emap.get(qid, {})
        domain = str(question.get("domain", ""))
        answer_format = str(question.get("answer_format", ""))
        answer = row.get("answer", "")
        by_domain[domain][answer] += 1
        by_format[answer_format][answer] += 1
        reviewed[(domain, str(bool(evidence.get("reviewed"))))] += 1

        initial_parsed = extract_json_object(evidence.get("raw_model_output") or "")
        review_parsed = extract_json_object(evidence.get("raw_review_output") or "")
        use_review = args.analysis_stage == "final" and bool(evidence.get("reviewed")) and bool(review_parsed)
        parsed = review_parsed if use_review else initial_parsed
        evaluated_answer = (
            normalize_answer(answer, answer_format or "multi")
            if args.analysis_stage == "final"
            else normalize_answer(str(initial_parsed.get("answer", "")), answer_format or "multi")
        )
        reasons = review_risk_reasons(
            question,
            parsed,
            evaluated_answer,
            evidence.get("retrieved_evidence", []),
            bool(evidence.get("fallback_used")) if args.analysis_stage == "initial" else False,
            broad_multi_review=args.risk_scope == "broad",
        )
        if use_review:
            reasons = [
                reason
                for reason in reasons
                if reason not in {
                    "multi_answer_selects_three_or_more_options",
                    "tf_compound_or_cross_doc_statement",
                    "multi_requires_skeptical_review",
                }
            ]
        if args.analysis_stage == "final" and evidence.get("base_answer_fallback_used"):
            reasons.append("final_answer_fell_back_to_base")
        reasons = sorted(set(reasons))
        reason_counts.update(reasons)
        if reasons:
            suspects.append(
                {
                    "qid": qid,
                    "domain": domain,
                    "answer_format": answer_format,
                    "answer": answer,
                    "reviewed": str(bool(evidence.get("reviewed"))),
                    "analysis_source": "review" if use_review else "initial",
                    "risk_reasons": ";".join(reasons),
                }
            )

    suspect_output.parent.mkdir(parents=True, exist_ok=True)
    with suspect_output.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["qid", "domain", "answer_format", "answer", "reviewed", "analysis_source", "risk_reasons"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(suspects)

    print(f"answers: {len(rows)}")
    print(f"evidence: {len(evidence_rows)}")
    print(f"risk_scope: {args.risk_scope}")
    print(f"analysis_stage: {args.analysis_stage}")
    print(f"would_review_now: {len(suspects)}")
    print(f"suspect_output: {suspect_output}")
    print("reviewed_by_domain:", dict(reviewed))
    print("top_risk_reasons:", reason_counts.most_common(20))
    print("answer_distribution_by_domain:")
    for domain, counter in sorted(by_domain.items()):
        print(f"  {domain}: {counter.most_common()}")
    print("answer_distribution_by_format:")
    for answer_format, counter in sorted(by_format.items()):
        print(f"  {answer_format}: {counter.most_common()}")
    if suspects:
        print("suspects_first_20:")
        for row in suspects[:20]:
            print(f"  {row['qid']} {row['answer']} {row['risk_reasons']}")


if __name__ == "__main__":
    main()
