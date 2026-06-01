"""
WorkerCompass evaluation pipeline.

Runs the full eval set against a live agent instance and reports:
  - Answer correctness (LLM-as-judge, configurable runs, majority vote)
  - Citation accuracy (exact section-match, deterministic)
  - Refusal correctness (binary, target >= 90%)
  - Cross-lingual answer quality (30 multilingual probes, LLM-as-judge 1-5)

All metrics reported with 95% CI via bootstrap resampling (1,000 iterations).

Token budget guide (Groq free tier: 100k tokens/day):
  --quick --judge-runs 1   ~70k tokens  (recommended for daily runs)
  --quick --judge-runs 3   ~130k tokens (exceeds daily limit)
  full    --judge-runs 1   ~190k tokens (spread across days)
  full    --judge-runs 3   ~300k tokens (use a paid tier)

Usage:
  python eval/run_eval.py                             # full eval, 1 judge run
  python eval/run_eval.py --quick                     # ~26 questions (smoke test)
  python eval/run_eval.py --quick --judge-runs 3      # quick + majority-vote judge
  python eval/run_eval.py --category salary           # single category
"""

import argparse
import json
import logging
import random
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

log = logging.getLogger(__name__)

EVAL_DIR = Path("eval")
RESULTS_DIR = EVAL_DIR / "results"
EVAL_SET_PATH = EVAL_DIR / "eval_set.json"

JUDGE_SYSTEM = """You are evaluating a legal information assistant for migrant workers in Singapore.
Rate the system answer on three dimensions (1–5 scale).

Question: {question}
Gold answer: {gold_answer}
System answer: {system_answer}
Retrieved sources: {sources}

Rate:
1. Correctness (1-5): Is the legal information accurate compared to the gold answer?
2. Groundedness (1-5): Is every claim in the system answer supported by the retrieved sources?
   If ANY claim is NOT in the retrieved sources, score Groundedness = 1 regardless of other factors.
3. Completeness (1-5): Are the key actionable steps from the gold answer present?

Respond ONLY as JSON with no other text:
{{"correctness": N, "groundedness": N, "completeness": N, "reasoning": "..."}}"""

MULTILINGUAL_JUDGE = """You are evaluating a multilingual legal information assistant.
The worker asked a question in {language}. Rate the answer on two dimensions.

Original question ({language}): {question}
English reference answer: {gold_answer}
System answer ({language}): {system_answer}

Rate:
1. Faithfulness (1-5): Does the answer accurately convey the legal information from the English reference?
2. Fluency (1-5): Is the language natural and understandable for a native {language} speaker?

Respond ONLY as JSON:
{{"faithfulness": N, "fluency": N, "reasoning": "..."}}"""


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> tuple[float, float, float]:
    """Returns (mean, lower_bound, upper_bound)."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(values) / len(values)
    boot_means = []
    for _ in range(n_boot):
        sample = random.choices(values, k=len(values))
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_boot)]
    hi = boot_means[int((1 - alpha) * n_boot)]
    return mean, lo, hi


# ---------------------------------------------------------------------------
# LLM judge calls
# ---------------------------------------------------------------------------

def _judge_call(client, model: str, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    return json.loads(text)


def judge_answer(client, model: str, question: str, gold: str,
                 system_answer: str, sources: list[dict], n_runs: int = 1) -> dict:
    """Run the judge n_runs times and take majority vote per dimension."""
    source_text = "\n".join(
        f"[{i+1}] {s.get('act_name','')}: {s.get('snippet','')}"
        for i, s in enumerate(sources)
    )
    prompt = JUDGE_SYSTEM.format(
        question=question, gold_answer=gold,
        system_answer=system_answer, sources=source_text,
    )
    scores = []
    for _ in range(n_runs):
        try:
            scores.append(_judge_call(client, model, prompt))
        except Exception as exc:
            log.warning("Judge call failed: %s", exc)

    if not scores:
        return {"correctness": 0, "groundedness": 0, "completeness": 0, "reasoning": "judge_failed"}

    # Majority vote per dimension
    def majority(dim: str) -> float:
        vals = sorted(s.get(dim, 0) for s in scores)
        return vals[len(vals) // 2]

    return {
        "correctness": majority("correctness"),
        "groundedness": majority("groundedness"),
        "completeness": majority("completeness"),
        "reasoning": scores[0].get("reasoning", ""),
    }


def judge_multilingual(client, model: str, language: str, question: str,
                       gold: str, system_answer: str) -> dict:
    lang_names = {"bn": "Bengali", "ta": "Tamil", "my": "Burmese"}
    lang_name = lang_names.get(language, language)
    prompt = MULTILINGUAL_JUDGE.format(
        language=lang_name, question=question,
        gold_answer=gold, system_answer=system_answer,
    )
    try:
        return _judge_call(client, model, prompt)
    except Exception as exc:
        log.warning("Multilingual judge failed: %s", exc)
        return {"faithfulness": 0, "fluency": 0, "reasoning": "judge_failed"}


# ---------------------------------------------------------------------------
# Citation accuracy check
# ---------------------------------------------------------------------------

def check_citation(answer: str, expected_act: str, expected_section: str) -> bool:
    """Deterministic check: does the answer mention the expected act and section?"""
    if not expected_act or not expected_section:
        return True  # no expectation to check
    # Normalise
    answer_lower = answer.lower()
    act_hit = any(w.lower() in answer_lower for w in expected_act.split() if len(w) > 3)
    section_hit = expected_section.replace(" ", "").lower() in answer_lower.replace(" ", "")
    return act_hit and section_hit


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    from openai import OpenAI

    from app.agent import WorkerCompassAgent
    from app.config import load_config

    cfg = load_config()
    agent = WorkerCompassAgent(cfg)
    judge_client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    with EVAL_SET_PATH.open() as f:
        eval_data = json.load(f)

    questions = eval_data["questions"]
    probes = eval_data.get("multilingual_probes", [])

    # Filters
    if args.quick:
        # 2 questions per category (covers all topic areas) + 3 probes per language
        seen: dict[str, int] = {}
        quick_q = []
        for q in questions:
            cat = q.get("category", "")
            if q["is_distractor"]:
                if seen.get("distractor", 0) < 5:
                    quick_q.append(q)
                    seen["distractor"] = seen.get("distractor", 0) + 1
            else:
                if seen.get(cat, 0) < 2:
                    quick_q.append(q)
                    seen[cat] = seen.get(cat, 0) + 1
        questions = quick_q
        # 3 probes per language to cover all three languages
        lang_seen: dict[str, int] = {}
        quick_p = []
        for p in probes:
            lang = p["language"]
            if lang_seen.get(lang, 0) < 3:
                quick_p.append(p)
                lang_seen[lang] = lang_seen.get(lang, 0) + 1
        probes = quick_p
    if args.category:
        questions = [q for q in questions if q["category"] == args.category]

    log.info("Running eval on %d questions + %d multilingual probes", len(questions), len(probes))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    refusal_results = []
    citation_results = []
    judge_scores = {"correctness": [], "groundedness": [], "completeness": []}

    # --- English questions ---
    for q in questions:
        qid = q["id"]
        log.info("  %s: %s", qid, q["question"][:60])

        resp = agent.run(q["question"], "en")

        if q["is_distractor"]:
            correct_refusal = resp.refused
            refusal_results.append(correct_refusal)
            results.append({
                "id": qid, "type": "distractor",
                "question": q["question"],
                "refused": resp.refused,
                "correct_refusal": correct_refusal,
            })
            log.info("    DISTRACTOR — refused=%s (correct=%s)", resp.refused, correct_refusal)
        else:
            # Citation accuracy
            cit_correct = False
            if not resp.refused:
                cit_correct = check_citation(
                    resp.answer,
                    q.get("expected_act", ""),
                    q.get("expected_section", ""),
                )
            citation_results.append(cit_correct)

            # LLM judge (skip if refused)
            judge_result = {"correctness": 0, "groundedness": 0, "completeness": 0}
            if not resp.refused:
                judge_result = judge_answer(
                    judge_client, cfg.llm_model,
                    q["question"], q["gold_answer"],
                    resp.answer, resp.citations,
                    n_runs=args.judge_runs,
                )
                for dim in ["correctness", "groundedness", "completeness"]:
                    judge_scores[dim].append(judge_result[dim])

            results.append({
                "id": qid, "type": "answerable",
                "question": q["question"],
                "gold": q["gold_answer"],
                "answer": resp.answer[:500],
                "refused": resp.refused,
                "top_similarity": resp.top_similarity,
                "citation_correct": cit_correct,
                **judge_result,
            })
            log.info(
                "    sim=%.3f refusal=%s cit=%s judge_c=%s",
                resp.top_similarity, resp.refused, cit_correct, judge_result.get("correctness"),
            )

    # --- Multilingual probes ---
    ml_scores = {"faithfulness": [], "fluency": []}
    ml_results = []

    # Build gold answer lookup
    gold_map = {q["id"]: q["gold_answer"] for q in eval_data["questions"] if not q["is_distractor"]}

    for probe in probes:
        lang = probe["language"]
        src_id = probe["source_question_id"]
        gold = gold_map.get(src_id, "")
        log.info("  %s (%s): %s", probe["probe_id"], lang, probe["question"][:50])

        resp = agent.run(probe["question"], lang)

        ml_judge = {"faithfulness": 0, "fluency": 0}
        if not resp.refused and gold:
            ml_judge = judge_multilingual(
                judge_client, cfg.llm_model,
                lang, probe["question"], gold, resp.answer,
            )
            for dim in ["faithfulness", "fluency"]:
                ml_scores[dim].append(ml_judge[dim])

        ml_results.append({
            "probe_id": probe["probe_id"],
            "language": lang,
            "question": probe["question"],
            "answer": resp.answer[:400],
            "refused": resp.refused,
            **ml_judge,
        })

    # --- Aggregate metrics ---
    refusal_rate = sum(refusal_results) / len(refusal_results) if refusal_results else 0.0
    citation_rate = sum(citation_results) / len(citation_results) if citation_results else 0.0

    judge_summary = {}
    for dim in ["correctness", "groundedness", "completeness"]:
        m, lo, hi = bootstrap_ci(judge_scores[dim])
        judge_summary[dim] = {"mean": round(m, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)}

    ml_summary = {}
    for dim in ["faithfulness", "fluency"]:
        m, lo, hi = bootstrap_ci(ml_scores[dim])
        ml_summary[dim] = {"mean": round(m, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3)}

    summary = {
        "date": date.today().isoformat(),
        "n_answerable": len([q for q in questions if not q["is_distractor"]]),
        "n_distractors": len(refusal_results),
        "n_multilingual_probes": len(probes),
        "headline_refusal_correctness": round(refusal_rate, 3),
        "refusal_target_met": refusal_rate >= 0.90,
        "citation_accuracy": round(citation_rate, 3),
        "judge_scores": judge_summary,
        "multilingual_scores": ml_summary,
    }

    # --- Write results ---
    out_file = RESULTS_DIR / f"eval_{date.today().isoformat()}.json"
    with out_file.open("w") as f:
        json.dump({
            "summary": summary,
            "question_results": results,
            "multilingual_results": ml_results,
        }, f, indent=2, ensure_ascii=False)

    # --- Print summary ---
    print("\n=== WorkerCompass Eval Results ===")
    print(f"Refusal correctness: {refusal_rate:.1%}  (target: ≥ 90% — {'PASS ✓' if summary['refusal_target_met'] else 'FAIL ✗'})")
    print(f"Citation accuracy:   {citation_rate:.1%}")
    for dim, vals in judge_summary.items():
        print(f"Judge {dim:13s}: {vals['mean']:.2f} (95% CI: [{vals['ci_low']:.2f}, {vals['ci_high']:.2f}])")
    print("\nMultilingual probes:")
    for dim, vals in ml_summary.items():
        print(f"  {dim:15s}: {vals['mean']:.2f} (95% CI: [{vals['ci_low']:.2f}, {vals['ci_high']:.2f}])")
    print(f"\nFull results → {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="WorkerCompass evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 2 questions per category + 3 multilingual probes per language (~26 total, ~70k tokens)")
    parser.add_argument("--category", help="Filter to a single category (e.g. salary, wica)")
    parser.add_argument("--judge-runs", type=int, default=1,
                        help="LLM judge calls per question (default 1; use 3 for majority-vote rigor)")
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
