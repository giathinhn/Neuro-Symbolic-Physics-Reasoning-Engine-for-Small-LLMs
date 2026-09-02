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

    cat1 = [] # None returned
    cat2 = [] # 0.0 returned
    cat3 = [] # Computed correct/near-correct but marked unverified
    cat4 = [] # Computed wrong number

    for item in failed_ns:
        v = item["neuro_symbolic"]["predicted_value"]
        u = item["neuro_symbolic"]["predicted_unit"]
        gt_v = item["ground_truth"]["value"]
        gt_u = item["ground_truth"]["unit"]
        ver = item["neuro_symbolic"]["is_verified"]

        if v is None:
            cat1.append(item)
        elif v == 0.0:
            cat2.append(item)
        elif not ver:
            cat3.append(item)
        else:
            cat4.append(item)

    print(f"\n1. Solver Returned None (Extraction / Underdetermined): {len(cat1)}")
    for it in cat1:
        print(f"   [{it['id']}] {it['question'][:70]}... | GT: {it['ground_truth']['value']} {it['ground_truth']['unit']}")
        print(f"       Eqs: {it['neuro_symbolic']['equations_used']}")

    print(f"\n2. Zero Value Returned (Implicit Boiling / Missing delta): {len(cat2)}")
    for it in cat2:
        print(f"   [{it['id']}] {it['question'][:70]}... | GT: {it['ground_truth']['value']} {it['ground_truth']['unit']}")
        print(f"       Eqs: {it['neuro_symbolic']['equations_used']}")

    print(f"\n3. Value Computed but NOT Verified (Verification False Rejection): {len(cat3)}")
    for it in cat3:
        v = it['neuro_symbolic']['predicted_value']
        u = it['neuro_symbolic']['predicted_unit']
        gt_v = it['ground_truth']['value']
        gt_u = it['ground_truth']['unit']
        print(f"   [{it['id']}] {it['question'][:70]}...")
        print(f"       GT: {gt_v} {gt_u} | NS: {v} {u} | Pure OK: {it['pure_llm']['is_correct']}")

    print(f"\n4. Wrong Number Computed (Model/Equation/Variable mismatch): {len(cat4)}")
    for it in cat4:
        v = it['neuro_symbolic']['predicted_value']
        u = it['neuro_symbolic']['predicted_unit']
        gt_v = it['ground_truth']['value']
        gt_u = it['ground_truth']['unit']
        print(f"   [{it['id']}] {it['question'][:70]}...")
        print(f"       GT: {gt_v} {gt_u} | NS: {v} {u} | Pure: {it['pure_llm']['predicted_value']} {it['pure_llm']['predicted_unit']} | Eqs: {it['neuro_symbolic']['equations_used']}")

if __name__ == "__main__":
    main()
