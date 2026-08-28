"""Run and compare 100-question quantitative physics benchmark: Pure LLM vs Neuro-Symbolic Engine."""

import json
import math
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("src"))
from physics_reasoning.core.config import load_config
from physics_reasoning.llm.provider import LiteLLMProvider
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator
from physics_reasoning.units.unit_engine import UnitEngine


def extract_pure_llm_answer(text: str) -> tuple[float | None, str | None]:
    """Extract numeric answer and unit from direct LLM text output."""
    if not text:
        return None, None
    
    # 1. Pattern like "Answer: 44.0 °C" or "Đáp án: 44.0 °C" or "44.0 °C"
    ans_match = re.search(r"(?:Answer|Đáp án|Kết quả|Đáp số)\s*[:=]\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z°/\^_\*%\s\(\)]+)?", text, re.IGNORECASE)
    if ans_match:
        try:
            val = float(ans_match.group(1))
            unit = ans_match.group(2).strip() if ans_match.group(2) else ""
            unit = unit.rstrip(".,;)\n")
            return val, unit
        except Exception:
            pass

    # 2. Look for numbers followed by unit symbols
    unit_pat = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(km/h|m/s\^2|m/s\*\*2|m/s|kg/m\^3|kg/m\*\*3|N/m\^3|N/m|J/\(kg\.K\)|J/kg\.K|kWh|kW|MW|MJ|kJ|Pa|N|J|W|kg|g|ohm|Ω|V|A|min|s|h|km|m|cm|dm|l|lit|°C|K)"
    matches = list(re.finditer(unit_pat, text, re.IGNORECASE))
    if matches:
        last_m = matches[-1]
        try:
            val = float(last_m.group(1))
            unit = last_m.group(2).strip()
            return val, unit
        except Exception:
            pass

    # 3. Fallback: extract last float in text
    floats = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if floats:
        try:
            return float(floats[-1]), None
        except Exception:
            pass

    return None, None


def check_is_correct(
    pred_val: float | None,
    pred_unit: str | None,
    gt_val: float,
    gt_unit: str,
    unit_engine: UnitEngine,
    rel_tol: float = 0.03,
) -> tuple[bool, str]:
    """Check if predicted answer matches ground truth within tolerance, handling unit conversions."""
    if pred_val is None or not math.isfinite(pred_val):
        return False, "No numeric prediction"

    # Normalize units
    p_unit = (pred_unit or "").strip()
    g_unit = (gt_unit or "").strip()

    # Direct value check if units match or no unit
    if not p_unit or p_unit.lower() == g_unit.lower() or (p_unit in ("°C", "degC", "celsius", "độ C") and g_unit in ("°C", "degC", "celsius", "độ C")):
        if math.isclose(pred_val, gt_val, rel_tol=rel_tol, abs_tol=1e-3):
            return True, "Exact / Close match"

    # Try unit conversion to SI
    try:
        if p_unit and g_unit:
            conv = unit_engine.convert(pred_val, p_unit, g_unit)
            if math.isclose(conv.to_value, gt_val, rel_tol=rel_tol, abs_tol=1e-3):
                return True, f"Match after unit conversion ({p_unit} -> {g_unit})"
    except Exception:
        pass

    # Value close regardless of unit formatting
    if math.isclose(pred_val, gt_val, rel_tol=rel_tol, abs_tol=1e-3):
        return True, "Value match (unit unverified)"

    return False, f"Mismatch: pred={pred_val} {pred_unit}, expected={gt_val} {gt_unit}"


def main():
    bench_file = Path("data/problems/benchmark_100_quantitative.jsonl")
    if not bench_file.exists():
        print(f"Error: {bench_file} not found!")
        return

    problems = []
    with open(bench_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    print("=" * 80)
    print(f"STARTING 100-PROBLEM QUANTITATIVE BENCHMARK: PURE LLM vs NEURO-SYMBOLIC")
    print(f"Total problems: {len(problems)}")
    print("=" * 80)

    orchestrator = PipelineOrchestrator()
    pure_llm = orchestrator.llm
    unit_engine = UnitEngine()

    pure_correct = 0
    ns_correct = 0
    
    topic_stats = {}
    detailed_results = []

    pure_total_latency = 0.0
    ns_total_latency = 0.0
    pure_total_tokens = 0
    ns_total_tokens = 0

    start_time = time.perf_counter()

    for idx, p in enumerate(problems, 1):
        q_id = p["id"]
        topic = p["topic"]
        q_text = p["question"]
        gt_val = p["ground_truth_value"]
        gt_unit = p["ground_truth_unit"]

        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "pure_correct": 0, "ns_correct": 0}
        topic_stats[topic]["total"] += 1

        print(f"\n[{idx:03d}/{len(problems):03d}] [{topic.upper()}] {q_text}")

        # -------------------------------------------------------------
        # 1. EVALUATE MODE A: PURE LLM DIRECT
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        pure_resp_text = ""
        pure_tokens = 0
        try:
            resp = pure_llm.complete(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a physics problem solver. Solve the physics problem step-by-step and write the final answer in the exact format: 'Answer: <number> <unit>' at the end.",
                    },
                    {"role": "user", "content": q_text},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            pure_resp_text = resp.content
            pure_tokens = resp.usage.get("total_tokens", 0)
        except Exception as e:
            pure_resp_text = f"Error: {e}"

        t_pure = time.perf_counter() - t0
        pure_total_latency += t_pure
        pure_total_tokens += pure_tokens

        pure_val, pure_unit = extract_pure_llm_answer(pure_resp_text)
        is_pure_ok, pure_reason = check_is_correct(pure_val, pure_unit, gt_val, gt_unit, unit_engine)
        if is_pure_ok:
            pure_correct += 1
            topic_stats[topic]["pure_correct"] += 1

        # -------------------------------------------------------------
        # 2. EVALUATE MODE B: NEURO-SYMBOLIC PIPELINE
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        ns_sol = orchestrator.solve(q_text, problem_id=q_id)
        t_ns = time.perf_counter() - t0
        ns_total_latency += t_ns
        ns_total_tokens += ns_sol.total_tokens

        is_ns_ok, ns_reason = check_is_correct(ns_sol.answer_value, ns_sol.answer_unit, gt_val, gt_unit, unit_engine)
        if is_ns_ok and ns_sol.is_verified:
            ns_correct += 1
            topic_stats[topic]["ns_correct"] += 1

        # Print comparison line
        pure_status = "✅ PASS" if is_pure_ok else "❌ FAIL"
        ns_status = "✅ PASS" if (is_ns_ok and ns_sol.is_verified) else "❌ FAIL"
        print(f"  Expected       : {gt_val} {gt_unit}")
        print(f"  [Pure LLM]     : {pure_val} {pure_unit} -> {pure_status} ({t_pure:.2f}s, {pure_tokens} tok)")
        print(f"  [Neuro-Sym]    : {ns_sol.answer_value} {ns_sol.answer_unit} -> {ns_status} ({t_ns:.2f}s, {ns_sol.total_tokens} tok, attempts={ns_sol.num_attempts})")

        detailed_results.append({
            "id": q_id,
            "topic": topic,
            "question": q_text,
            "ground_truth": {"value": gt_val, "unit": gt_unit},
            "pure_llm": {
                "predicted_value": pure_val,
                "predicted_unit": pure_unit,
                "is_correct": is_pure_ok,
                "reason": pure_reason,
                "latency_s": round(t_pure, 3),
                "tokens": pure_tokens,
                "raw_output": pure_resp_text[:200],
            },
            "neuro_symbolic": {
                "predicted_value": ns_sol.answer_value,
                "predicted_unit": ns_sol.answer_unit,
                "is_correct": is_ns_ok and ns_sol.is_verified,
                "is_verified": ns_sol.is_verified,
                "num_attempts": ns_sol.num_attempts,
                "equations_used": ns_sol.equations_used,
                "reason": ns_reason,
                "latency_s": round(t_ns, 3),
                "tokens": ns_sol.total_tokens,
            },
        })

    total_bench_time = time.perf_counter() - start_time

    # -------------------------------------------------------------
    # 3. SUMMARY & REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON SUMMARY (100 PROBLEMS)")
    print("=" * 80)
    print(f"{'Domain / Topic':<20} | {'Total':<6} | {'Pure LLM (Pass %)':<20} | {'Neuro-Symbolic (Pass %)':<25}")
    print("-" * 80)
    for topic, stats in topic_stats.items():
        tot = stats["total"]
        p_pct = (stats["pure_correct"] / tot) * 100
        ns_pct = (stats["ns_correct"] / tot) * 100
        print(f"{topic:<20} | {tot:<6} | {stats['pure_correct']}/{tot} ({p_pct:5.1f}%)         | {stats['ns_correct']}/{tot} ({ns_pct:5.1f}%)")

    total_pure_pct = (pure_correct / len(problems)) * 100
    total_ns_pct = (ns_correct / len(problems)) * 100
    print("-" * 80)
    print(f"{'OVERALL TOTAL':<20} | {len(problems):<6} | {pure_correct}/{len(problems)} ({total_pure_pct:5.1f}%)        | {ns_correct}/{len(problems)} ({total_ns_pct:5.1f}%)")
    print("=" * 80)
    print(f"Pure LLM Avg Latency       : {pure_total_latency / len(problems):.2f}s | Avg Tokens: {pure_total_tokens / len(problems):.1f}")
    print(f"Neuro-Symbolic Avg Latency : {ns_total_latency / len(problems):.2f}s | Avg Tokens: {ns_total_tokens / len(problems):.1f}")
    print(f"Total Benchmark Run Time   : {total_bench_time:.2f}s")
    print("=" * 80)

    # Save results to JSON
    out_path = Path("data/problems/benchmark_100_comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_problems": len(problems),
                "pure_llm_accuracy": total_pure_pct,
                "neuro_symbolic_accuracy": total_ns_pct,
                "pure_llm_correct": pure_correct,
                "neuro_symbolic_correct": ns_correct,
                "topic_breakdown": topic_stats,
                "pure_avg_latency_s": pure_total_latency / len(problems),
                "ns_avg_latency_s": ns_total_latency / len(problems),
                "total_runtime_s": total_bench_time,
            },
            "details": detailed_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"Detailed comparison saved to: {out_path}")


if __name__ == "__main__":
    main()
