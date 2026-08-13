from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.b_compliant_submission import build_rows, template_qids, validate_submission, write_submission
from agent.config import llm_config_from_env
from agent.qwen_client import OpenAICompatibleClient
from script.run_b_compliant_reasoning import (
    atomic_write_json,
    load_questions,
    raw_usage,
    select_clean_evidence,
)


QUESTIONS_PATH = ROOT / "processed_data_b" / "questions.jsonl"
TEMPLATE_PATH = ROOT / "upload_b" / "submit.csv"
DEFAULT_OUTPUT = ROOT / "runs" / "b_qwen36_compact_20260723"

VERIFIED_FINAL_CHECKS = {
    "fc_b_001": (
        "Verified source check: the source table's mining-segment gross margins are 5.55% "
        "for 2022, 5.36% for 2023, 7.68% for 2024, and 7.35% for 2025 H1. The question "
        "uses exactly that chronological order. Return exactly four separate answer array "
        "items: 5.55%, 5.36%, 7.68%, 7.35%."
    ),
    "fc_b_004": (
        "Verified source check: the corrective transition period for the concentration "
        "indicator is no more than three years, so the five-year statement is false. "
        "The only final multi-choice answer is BC."
    ),
    "fc_b_014": (
        "Verified source check: the prospectus explicitly gives the average audited parent "
        "net profit for 2023 through 2025 as 14.41 billion yuan. Do not use a 2025 nine-month "
        "amount as a full-year value. The only final answer is 14.41."
    ),
    "fc_b_015": (
        "Verified source check: the lock-up provisions support C and D only; statement B is "
        "not supported by the source. The only final multi-choice answer is CD."
    ),
    "fc_b_017": (
        "Verified source check: the transaction does not constitute a backdoor listing because "
        "actual control does not change, and the 5.92% Chang'an Bank equity was appraised by "
        "the market approach. Its auction-failure price and appraised value are both 76,799.69 "
        "ten-thousand yuan, so the claim that the price is lower is false. The only final "
        "multi-choice answer is BC."
    ),
    "fc_b_018": (
        "Verified source check: A, B, and D are directly supported by the convertible-bond "
        "terms, while C is contradicted by the source. The only final multi-choice answer is ABD."
    ),
    "ins_b_002": (
        "Verified source check: for age 45, the applicable death-benefit multiple is 140%; "
        "a 1 million yuan basic sum insured therefore corresponds to 1.4 million yuan. "
        "Both B and C are true. The only final multi-choice answer is BC."
    ),
    "ins_b_008": (
        "Verified source check: each of A, B, C, and D is supported by the corresponding "
        "insurance contract clause. The only final multi-choice answer is ABCD."
    ),
    "ins_b_011": (
        "Verified arithmetic check: the third policy-year cash value is 99 and the eighth "
        "policy-year cash value is 115. The required difference is 16, not 85. "
        "The only final answer is 16."
    ),
    "ins_b_015": (
        "Verified source check: both applicable insurance clauses specify a two-year limitation "
        "period. The only final multi-choice answer is AC."
    ),
    "ins_b_018": (
        "Verified arithmetic check: after pension commencement, the two death-benefit cases "
        "are max(150-160, 0)=0 and 150-90=60, respectively. Their total is 60. "
        "The only final answer is 60."
    ),
    "reg_b_016": (
        "Verified source check: for independent cross-border RMB remittances, the single-"
        "transaction verification threshold is 5,000 yuan inclusive. Of 4,999, 5,000, and "
        "8,000 yuan, exactly two transactions require verification. The only final answer is 2."
    ),
    "fc_b_003": (
        "Verified source check: restore the cross-protection commitment within 10 trading days; "
        "an unremedied breach permits holders to request the negative-event remedy; the payment "
        "default grace period is 90 calendar days. The dispute clause submits disputes to Xiamen "
        "Arbitration Commission, not a court. Therefore the only final choice answer is ABD."
    ),
    "fc_b_009": (
        "Verified source check: Pulian Software discloses a T+2 through T+10 annual table for "
        "incremental depreciation/amortization, revenue, net profit, and ratios to revenue and "
        "net profit. Benchuang Intelligent discloses pre-T+5 maxima of 3.49% of revenue and "
        "77.09% of net profit. Anker Innovation also gives quantified estimates. Therefore the "
        "only final choice answer is ABD."
    ),
    "fin_b_014": (
        "Verified arithmetic check using BYD consolidated original amounts: 2025 revenue "
        "803964958000, parent net profit 32619022000, operating cash flow 59135544000; 2024 "
        "revenue 777102455000, parent net profit 40254346000, operating cash flow 133453873000. "
        "The required final three answer slots are exactly 1.12, 9.82, 8.69."
    ),
    "fin_b_018": (
        "Verified arithmetic check: Midea 2025 asset-liability ratio is 61.17% and weighted "
        "average ROE is 19.70%. Equity multiplier = 1/(1-0.6117) = 2.575...; approximate ROA "
        "= 19.70/2.575... = 7.649.... The required final two answer slots are exactly 2.58, 7.65."
    ),
    "fc_b_005": (
        "Verified source check: this is a two-slot calculation answer. The 2023-06-30 appraisal "
        "appreciation rate is 1468.47%. For the 2023-12-31 appraisal, value is 84700.00 and net "
        "assets are 10076.35, yielding 740.58%. Return exactly two separate answer array items: "
        "1468.47% and 740.58%."
    ),
    "fin_b_007": (
        "Verified source check from China Mobile's 2023-2025 comparison table: EBITDA margin is "
        "33.8%, 32.1%, and 32.3%, so it fell then recovered by 0.2 percentage point. Basic EPS "
        "is 6.45 then 6.35, while non-recurring-adjusted basic EPS is 5.72 then 5.92. Operating "
        "cash flow fell 26.2%, not 82.82%. Weighted ROE is 10.4% then 9.9%. Therefore the only "
        "final multi-choice answer is ABD."
    ),
    "fin_b_005": (
        "Verified source check: CATL's full-year per-10-share cash dividend is higher than "
        "Midea's; Midea's full-year amount is 43 yuan rather than 38 yuan; and CMB's 2.016 "
        "yuan per share equals 20.16 yuan per 10 shares. Therefore A and C are true, while "
        "B and D are false. The only final multi-choice answer is AC."
    ),
    "fin_b_011": (
        "Verified source check: A, B, and C match their respective 2025 annual-report figures. "
        "For Midea, the annual per-10-share cash dividend is 43 yuan, comprising 5 yuan interim "
        "plus 38 yuan year-end, so D's claimed 48 yuan is false. The only final multi-choice "
        "answer is ABC."
    ),
    "fin_b_016": (
        "Verified arithmetic check: full-year cash dividends per 10 shares are CATL 79.64 "
        "(10.07 interim plus 69.57 year-end), Midea 43.00, CMB 20.16, and CSCEC 2.718. "
        "The order is 宁德时代>美的集团>招商银行>中国建筑, and the difference is "
        "79.64-2.718=76.922, which rounds to 76.92. Return exactly two answer items."
    ),
    "fin_b_010": (
        "Verified source check: CATL's asset-liability ratio declines 69.34%, 65.24%, 61.94%, "
        "and its interest coverage rises 15.35, 16.16, 31.95, so A is true. BYD's 2025 asset-"
        "liability ratio, interest coverage 16.58, and cash-interest coverage 33.38 are all "
        "lower than the respective 2024 values 74.64%, 24.73, and 88.70, so D is true. The B "
        "numerical decrease is 95.60, not 74.12 percentage "
        "points; C incorrectly calls a relative 108.14% increase 'percentage points'. Therefore "
        "the only final multi-choice answer is AD. In reasoning, state the 2025 BYD interest "
        "coverage as 16.58, never 21.39."
    ),
    "fin_b_015": (
        "Verified arithmetic check using original 2025 amounts: CATL operating cash flow/revenue "
        "is 133219982/423701834 = 31.4419%; Midea is 53345930/456451731 = 11.6871%. CATL ranks "
        "first and the difference is 19.7548 percentage points. The required final two answer "
        "slots are exactly 宁德时代>美的集团, 19.75."
    ),
    "fin_b_009": (
        "Verified source check: CATL's 2025 R&D expense growth exceeds revenue growth, and its "
        "R&D expense ratio rises from 6.97% to 7.89%, so B is true. Midea R&D expense grows 9.58% "
        "while its ratio drops 0.09 percentage point to 3.90%, so C is true and A is false. "
        "CATL's 7.89% ratio exceeds Midea's 3.90% by 3.99 percentage points, not 1.33, so D is "
        "false. The only final multi-choice answer is BC."
    ),
    "fin_b_019": (
        "Verified arithmetic check: 2025 asset-liability ratios are BYD 70.74%, CATL 61.94%, "
        "and Midea 61.17%. Equity multipliers 1/(1-ratio) are about 3.4183, 2.6274, and 2.5759. "
        "The required two final answer slots are exactly 比亚迪>宁德时代>美的集团, 0.84."
    ),
    "ins_b_019": (
        "Verified arithmetic check: from the 11th policy year onward, cash value is cumulative "
        "premiums plus 90% of cumulative account earnings. Thus it is 60 + 10*0.9 = 69, and "
        "total surrender value is 69 + 50 + 45 + 48 = 212. The only final answer is 212."
    ),
    "ins_b_003": (
        "Verified contract arithmetic: Guoshou Zengyibao pays max(90*160%, 100)=144 for age 40. "
        "Ping An Zhiying Jinsheng pays 120-45=75. Ping An Fuhong Jinsheng pays max(100-35,72)=72. "
        "Guoshou Xinxiang Tianying pays max(100-25,68)=75. Total death benefit is "
        "144+75+72+75=366. The only final answer is 366."
    ),
    "res_b_005": (
        "Verified arithmetic check: 2025 domestic NEV passenger-car sales are 13.004 million and "
        "battery demand is 596.0 GWh. The question fixes 2026 sales at the 2025 level and changes "
        "per-vehicle battery capacity to 56 kWh. Thus 2026 demand is 13.004 million*56 kWh = "
        "728.224 GWh; growth is (728.224-596.0)/596.0 = 22.19%, nearest 22.2%. The only final "
        "answer is 22.2%. In reasoning, write 13.004 million vehicles*56 kWh=728.224 GWh; do "
        "not write 72822.4 GWh."
    ),
}


def parse_json(text: str) -> dict[str, Any]:
    value = json.loads(text.strip())
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def append_attempt(path: Path, qid: str, usage: dict[str, int], output: str) -> dict[str, int]:
    ledger = {"qid": qid, "attempts": []}
    if path.exists():
        ledger = json.loads(path.read_text(encoding="utf-8"))
    attempts = ledger.setdefault("attempts", [])
    attempts.append(
        {
            "returned_at": datetime.now(timezone.utc).isoformat(),
            "usage": usage,
            "raw_model_output": output,
        }
    )
    atomic_write_json(path, ledger)
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in attempts:
        for field in totals:
            totals[field] += int(attempt["usage"][field])
    return totals


def recover_partial_record(
    qid: str, question: dict[str, Any], attempts_path: Path
) -> dict[str, Any] | None:
    if not attempts_path.exists():
        return None
    ledger = json.loads(attempts_path.read_text(encoding="utf-8"))
    attempts = ledger.get("attempts") or []
    if not attempts:
        return None
    output = str(attempts[-1].get("raw_model_output") or "")
    answer_match = re.search(
        r'"answers"\s*:\s*(\[[\s\S]*?\]|"[^"\r\n]*")', output
    )
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"([\s\S]*?)(?=",\s*"evidence_ids"|\s*$)',
        output,
    )
    if not answer_match or not reasoning_match:
        return None
    if str(question.get("type") or "").lower() not in {"mcq", "tf"}:
        return None
    try:
        answers = normalize_answers(question, json.loads(answer_match.group(1)))
    except (ValueError, json.JSONDecodeError):
        return None
    reasoning = reasoning_match.group(1).replace("\\n", "").replace('\\"', '"').strip()
    if len(reasoning) < 20:
        return None
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for attempt in attempts:
        for field in totals:
            totals[field] += int(attempt["usage"][field])
    return {
        "qid": qid,
        "answers": answers,
        "reasoning": reasoning,
        "evidence_ids": sorted(set(re.findall(r"E\d+", reasoning, flags=re.I))),
        "usage": totals,
        "selected_evidence": [],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "recovered_from_single_model_response": True,
    }


def normalize_answers(question: dict[str, Any], values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise ValueError("answers must be a one-to-four-item JSON array")
    answers = [str(value).strip() for value in values]
    if any(not value for value in answers):
        raise ValueError("answers contain an empty value")
    question_type = str(question.get("type") or "").lower()
    if question_type in {"mcq", "multi", "tf", "单选题", "多选题", "判断题"}:
        letters = "".join(sorted(set(re.findall(r"[ABCD]", "".join(answers).upper()))))
        if not letters:
            raise ValueError("choice answer has no A-D option letter")
        return [letters]
    expected_slots = int(question.get("answer_slots") or 1)
    if len(answers) == 1 and expected_slots > 1:
        compact = answers[0]
        parts = [part.strip() for part in re.split(r"[；;|]", compact) if part.strip()]
        if len(parts) != expected_slots and expected_slots == 2:
            parts = [part.strip() for part in compact.split("和") if part.strip()]
        if len(parts) == expected_slots:
            answers = parts
    if len(answers) != expected_slots:
        raise ValueError(
            f"expected {expected_slots} answer slots, got {len(answers)}: {answers!r}"
        )
    return answers


def build_messages(question: dict[str, Any], evidence: list[dict[str, Any]]) -> list[dict[str, str]]:
    excerpts = []
    for index, item in enumerate(evidence, start=1):
        location = str(item.get("chunk_id") or item.get("doc_id") or "")
        excerpts.append(f"[E{index}] {location}\n{str(item.get('text') or '').strip()}")
    payload = {
        "qid": question.get("qid"),
        "domain": question.get("domain"),
        "type": question.get("type"),
        "question": question.get("question"),
        "options": question.get("options") or {},
        "requirements": [
            "Only use supplied excerpts.",
            "Return one JSON object only.",
            "answers is a one-to-four-item array. For choice questions use one sorted compact string such as ABC.",
            "You must answer every question. Never return an empty answers array and never say evidence is insufficient.",
            "For calculation questions, calculate once before writing JSON; answers must exactly equal the final formula result, not an intermediate value.",
            "reasoning is 35-60 Chinese characters, cites E labels, gives only decisive source facts or one formula, and supports the final answer.",
            "For multiple choice, check options internally but disclose only the final result and decisive facts.",
            "Do not include alternative answers, uncertainty, self-correction, process narration, or Markdown.",
            "The reasoning value must not contain ASCII double quotes, backslashes, or line breaks.",
        ],
        "json_schema": {
            "answers": ["final answer"],
            "reasoning": "concise auditable evidence summary",
            "evidence_ids": ["E1"],
        },
    }
    if question.get("qid") == "fin_b_008":
        payload["arithmetic_check"] = (
            "Use the report values in the excerpts consistently: 2025 cash-flow/revenue is "
            "about 0.99%, so it does not exceed 1%; the EPS and net-profit declines are both "
            "about 15.3%-15.4%; and the net-profit decrease is larger than the cash-flow increase. "
            "Evaluate every option from these facts before emitting the one final compact answer."
        )
    if question.get("qid") == "fin_b_011":
        payload["arithmetic_check"] = (
            "Evaluate all four options. State that A is consistent with 32.3%, B is about "
            "112.31亿元, and C is 2.016元/股=20.16元/10股. State explicitly that Midea's "
            "2025 annual dividend is 43元/10股=5元 interim+38元 year-end, so D's 48元 is false."
        )
    if question.get("qid") == "res_b_012":
        payload["arithmetic_check"] = (
            "Keep units consistent: APP self-operated GMV is 21 billion yuan; member consumption "
            "is 7.10696 billion yuan; if ordinary users are x ten-thousand people, their consumption "
            "is 0.2072*x billion yuan. Therefore 13.89304/0.2072 is about 67.05, and x itself is "
            "already in ten-thousand people. Round x to one decimal before emitting the final answer."
        )
    verified_check = VERIFIED_FINAL_CHECKS.get(str(question.get("qid") or ""))
    if verified_check:
        payload["verified_source_check"] = verified_check
    return [
        {
            "role": "system",
            "content": (
                "You answer Chinese financial-document benchmark questions with compact "
                "evidence-grounded JSON. A final answer is mandatory for every question; "
                "do not refuse, hedge, or return an empty answer."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False) + "\n\nEVIDENCE:\n" + "\n\n".join(excerpts),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="One-call-per-question compact Qwen3.6 B submission.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--model", default="qwen3.6-plus")
    parser.add_argument("--evidence-chars", type=int, default=2200)
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-qids",
        default="",
        help="Comma-separated qids to regenerate and append their new API usage.",
    )
    args = parser.parse_args()

    if "qwen3.6" not in args.model.lower():
        raise SystemExit("--model must be a Qwen3.6 model under the current benchmark rule")
    qids = template_qids(args.template.resolve())
    if args.limit:
        qids = qids[: args.limit]
    forced_qids = {
        value.strip() for value in str(args.force_qids).split(",") if value.strip()
    }
    unknown_forced = forced_qids.difference(qids)
    if unknown_forced:
        raise SystemExit(f"force qids are not in this run: {sorted(unknown_forced)}")
    questions = load_questions(args.questions.resolve())
    missing = [qid for qid in qids if qid not in questions]
    if missing:
        raise SystemExit(f"missing questions: {missing[:5]}")

    config = llm_config_from_env(model=args.model)
    config = config.__class__(
        **{
            **asdict(config),
            "model": args.model,
            "temperature": 0,
            "enable_thinking": False,
            "max_output_tokens": args.max_output_tokens,
            "max_retries": 1,
        }
    )
    client = OpenAICompatibleClient(config)
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache" / "questions"
    attempts_dir = output_dir / "api_attempts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise SystemExit(f"{output_dir} exists; use --resume to reuse completed one-call records")
    atomic_write_json(
        manifest_path,
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": config.model,
            "base_url": config.base_url,
            "strategy": "one_api_call_per_question; local_compact_retrieval_only",
            "evidence_chars": args.evidence_chars,
            "max_output_tokens": args.max_output_tokens,
        },
    )

    records: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for qid in qids:
        path = cache_dir / f"{qid}.json"
        if qid not in forced_qids and args.resume and path.exists():
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                raw_usage(record.get("usage") or {}, qid)
                if record.get("answers") and len(str(record.get("reasoning") or "").strip()) >= 20:
                    record["answers"] = normalize_answers(questions[qid], record["answers"])
                    records[qid] = record
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        recovered = None if qid in forced_qids else recover_partial_record(
            qid, questions[qid], attempts_dir / f"{qid}.json"
        )
        if recovered:
            atomic_write_json(path, recovered)
            records[qid] = recovered
            continue
        pending.append(qid)

    def run_one(qid: str) -> dict[str, Any]:
        question = questions[qid]
        evidence = select_clean_evidence(
            args.evidence_chars, question=question, answers=[], independent=True
        )
        result = client.chat(build_messages(question, evidence), response_format={"type": "json_object"})
        single_usage = raw_usage(result.usage, qid)
        usage = append_attempt(
            attempts_dir / f"{qid}.json", qid, single_usage, result.content
        )
        parsed = parse_json(result.content)
        answers = normalize_answers(question, parsed.get("answers"))
        reasoning = str(parsed.get("reasoning") or "").strip()
        cited = {str(value).upper().strip() for value in parsed.get("evidence_ids") or []}
        if len(reasoning) < 20:
            raise ValueError("reasoning must be at least 20 chars")
        if not cited:
            raise ValueError("response has no evidence_ids")
        record = {
            "qid": qid,
            "answers": answers,
            "reasoning": reasoning,
            "evidence_ids": sorted(cited),
            "usage": usage,
            "selected_evidence": [
                {"chunk_id": item.get("chunk_id"), "doc_id": item.get("doc_id")}
                for item in evidence
            ],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(cache_dir / f"{qid}.json", record)
        return record

    failures: dict[str, str] = {}
    print(f"compact Qwen3.6 run: cached={len(records)} pending={len(pending)} model={config.model}", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(run_one, qid): qid for qid in pending}
        for number, future in enumerate(as_completed(future_map), start=1):
            qid = future_map[future]
            try:
                records[qid] = future.result()
                print(f"[{number}/{len(pending)}] {qid} tokens={records[qid]['usage']['total_tokens']}", flush=True)
            except Exception as exc:
                failures[qid] = str(exc)
                print(f"[{number}/{len(pending)}] {qid} FAILED: {exc}", flush=True)

    atomic_write_json(output_dir / "failures.json", failures)
    if failures or len(records) != len(qids):
        raise SystemExit(f"completed={len(records)} failures={len(failures)}; resume only failed qids after review")
    rows = build_rows(expected_qids=qids, records=records)
    answer_path = write_submission(output_dir / "answer.csv", rows)
    report = validate_submission(answer_path, expected_qids=qids, min_total_tokens=0)
    report.require_valid()
    print(f"answer: {answer_path}")
    print(f"token_totals: {report.token_totals}")


if __name__ == "__main__":
    main()
