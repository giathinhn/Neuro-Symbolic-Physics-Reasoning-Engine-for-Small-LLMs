import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("src"))

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator
from physics_reasoning.llm.output_parser import parse_llm_output

orchestrator = PipelineOrchestrator()
p_text = "Đặt một hiệu điện thế U = 24 V vào hai đầu điện trở R = 12 ohm. Tính cường độ dòng điện chạy qua điện trở."

print("Problem:", p_text)
# Let's see what the LLM generates
resp = orchestrator.llm.complete(
    messages=[
        {"role": "system", "content": "You are a physics solver. Respond with JSON."},
        {"role": "user", "content": p_text},
    ]
)
print("\n--- RAW LLM RESPONSE ---")
print(resp.content)

print("\n--- PARSED OUTPUT ---")
parsed = parse_llm_output(resp.content)
print(parsed)

print("\n--- EXTRACTED QUANTITIES ---")
merged = orchestrator.quantity_extractor.merge_with_text_quantities(parsed.quantities, p_text)
st_quantities = orchestrator.quantity_extractor.standardize_all(merged)
for q in st_quantities:
    print(q)

print("\n--- SOLVER RUN ---")
sol = orchestrator.solve(p_text)
print("sol:", sol)
