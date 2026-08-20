import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

# Đổi câu hỏi của bạn tại đây
question = "Tại sao cánh quạt bị bám bụi sau 1 thời gian sử dụng."

solution = orchestrator.solve(question)

print("Loại bài toán:", "Định tính" if solution.is_qualitative else "Định lượng")
print("Trạng thái xác thực:", "Thành công" if solution.is_verified else "Chưa xác thực")

if solution.is_qualitative:
    print("Kết luận giải thích:", solution.final_explanation)
else:
    print(f"Đáp án: {solution.answer_value} {solution.answer_unit}")