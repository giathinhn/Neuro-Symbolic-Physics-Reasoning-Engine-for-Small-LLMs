import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()
question = "Tại sao cánh quạt bị bám bụi sau 1 thời gian sử dụng."

solution = orchestrator.solve(question)

print("Loại bài toán:", "Định tính" if solution.is_qualitative else "Định lượng")
print("Trạng thái xác thực:", "Thành công" if solution.is_verified else "Chưa xác thực")
print("Error message:", solution.error_message)
print("Attempts:", solution.num_attempts)
print("Total LLM calls:", solution.total_llm_calls)
print("Steps:")
for s in solution.solve_steps:
    print(f"  Step {s.step_number}: {s.action} -> {s.output_data}")

if solution.is_qualitative:
    print("Kết luận giải thích:", solution.final_explanation)
    if solution.qualitative_output:
        print("Parsed JSON:", solution.qualitative_output.model_dump_json(indent=2))
