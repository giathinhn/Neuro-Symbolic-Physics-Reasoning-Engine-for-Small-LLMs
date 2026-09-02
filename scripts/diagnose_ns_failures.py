import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    with open("data/problems/benchmark_100_comparison_results.json", encoding="utf-8") as f:
        data = json.load(f)

    details = data["details"]
    failed_ns = [item for item in details if not item["neuro_symbolic"]["is_correct"]]

    print(f"Total Failed Neuro-Symbolic Cases: {len(failed_ns)} / {len(details)}")

    categories = {}
    for item in failed_ns:
        q_id = item["id"]
        topic = item["topic"]
        q = item["question"]
        gt = item["ground_truth"]
        ns = item["neuro_symbolic"]
        pure = item["pure_llm"]

        pred_val = ns["predicted_value"]
        pred_unit = ns["predicted_unit"]
        gt_val = gt["value"]
        gt_unit = gt["unit"]
        eqs = ns.get("equations_used", [])
        reason = ns.get("reason", "")
        attempts = ns.get("num_attempts", 0)

        # Analyze root cause
        if pred_val is None:
            cat = "1. Solver could not resolve (None returned)"
        elif pred_val == 0.0:
            cat = "2. Zero value (Missing implicit parameter or cancellation)"
        elif not ns.get("is_verified", False):
            cat = "3. Value computed but NOT verified (Verification rejection)"
        else:
            cat = "4. Value computed & verified, but wrong magnitude (Wrong equation / model)"

        categories.setdefault(cat, []).append({
            "id": q_id,
            "topic": topic,
            "q": q,
            "gt": f"{gt_val} {gt_unit}",
            "pred": f"{pred_val} {pred_unit}",
            "pure_pred": f"{pure['predicted_value']} {pure['predicted_unit']}",
            "pure_ok": pure["is_correct"],
            "eqs": eqs,
            "reason": reason,
            "attempts": attempts,
        })

    for cat_name, items in sorted(categories.items()):
        print(f"\n{'='*80}")
        print(f"=== {cat_name} ({len(items)} cases) ===")
        print(f"{'='*80}")
        for it in items:
            print(f"[{it['id']}] ({it['topic']}) {it['q']}")
            print(f"    GT: {it['gt']} | NS: {it['pred']} | Pure: {it['pure_pred']} (Pure OK={it['pure_ok']})")
            print(f"    Eqs: {it['eqs']} | Reason: {it['reason']} | Attempts: {it['attempts']}")
            print("-" * 60)

if __name__ == "__main__":
    main()
