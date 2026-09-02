import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    path = Path("data/problems/benchmark_100_comparison_results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    details = data["details"]
    both_pass = []
    ns_only = []
    pure_only = []
    both_fail = []

    for item in details:
        p_ok = item["pure_llm"]["is_correct"]
        ns_ok = item["neuro_symbolic"]["is_correct"]
        if p_ok and ns_ok:
            both_pass.append(item)
        elif ns_ok and not p_ok:
            ns_only.append(item)
        elif p_ok and not ns_ok:
            pure_only.append(item)
        else:
            both_fail.append(item)

    print(f"Total: {len(details)}")
    print(f"Both Passed: {len(both_pass)}")
    print(f"Neuro-Symbolic ONLY Passed (NS Outperformed LLM): {len(ns_only)}")
    print(f"Pure LLM ONLY Passed: {len(pure_only)}")
    print(f"Both Failed: {len(both_fail)}")

    print("\n" + "="*80)
    print("1. CASES WHERE NEURO-SYMBOLIC WON (Pure LLM Failed due to math/units/hallucinations)")
    print("="*80)
    for item in ns_only:
        print(f"ID: {item['id']} ({item['topic']})")
        print(f"Question : {item['question']}")
        print(f"Expected : {item['ground_truth']['value']} {item['ground_truth']['unit']}")
        print(f"Pure LLM : {item['pure_llm']['predicted_value']} {item['pure_llm']['predicted_unit']} (Output snippet: {repr(item['pure_llm']['raw_output'][:90])})")
        print(f"Neuro-Sym: {item['neuro_symbolic']['predicted_value']} {item['neuro_symbolic']['predicted_unit']} (Equations: {item['neuro_symbolic']['equations_used']})")
        print("-" * 80)

    print("\n" + "="*80)
    print("2. FAILURE CLUSTERS IN NEURO-SYMBOLIC (Why Pure LLM answered right while NS failed)")
    print("="*80)
    reasons = {}
    for item in pure_only:
        ns = item["neuro_symbolic"]
        p_val = ns["predicted_value"]
        p_unit = ns["predicted_unit"]
        gt_val = item["ground_truth"]["value"]
        
        # Categorize
        if p_val is None:
            cat = "Extraction / Parsing Failure (None returned)"
        elif not ns["is_verified"]:
            cat = "Verification / Self-Repair Rejection (Computed value failed verification check)"
        elif p_val == 0.0:
            cat = "Zero / Implicit Parameter Missing (e.g. boiling temp 100C or implicit delta)"
        else:
            cat = "Formula or Multi-step Indexing Mismatch (e.g. 3-resistor circuit, partial efficiency)"
        
        reasons[cat] = reasons.get(cat, 0) + 1

    for cat, count in reasons.items():
        print(f"  * {cat}: {count} cases")

if __name__ == "__main__":
    main()
