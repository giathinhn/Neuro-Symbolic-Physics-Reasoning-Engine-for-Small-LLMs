import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("src"))

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
p_text = "Một vật rơi tự do từ độ cao 45 m xuống đất. Lấy g = 10 m/s^2. Tính thời gian rơi của vật."

print("=== 1. Text extracted quantities ===")
tq = orchestrator.quantity_extractor.extract_quantities_from_text(p_text)
for q in tq:
    print(f"  sym={q.symbol}, val={q.value}, unit={q.unit}, role={q.role}")

print("\n=== 2. Relevant equations ===")
reqs = orchestrator.retriever.retrieve_for_problem(p_text)
for eq in reqs:
    print(f"  {eq.id}: {eq.expression}")

print("\n=== 3. LLM call & parsing ===")
messages = [
    {"role": "system", "content": orchestrator._solve_quantitative.__code__.co_consts},
]
sol = orchestrator.solve(p_text)
print("sol.answer_value:", sol.answer_value)
print("sol.answer_unit:", sol.answer_unit)
print("sol.is_verified:", sol.is_verified)
print("sol.error_message:", sol.error_message)
print("sol.equations_used:", [eq.expression for eq in sol.equations_used])
print("sol.quantities:", [(q.symbol, q.value, q.unit) for q in sol.quantities])
