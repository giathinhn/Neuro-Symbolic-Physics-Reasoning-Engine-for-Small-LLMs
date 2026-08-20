import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

# Đổi câu hỏi của bạn tại đây
question = "Tại sao cánh quạt bị bám bụi sau 1 thời gian sử dụng."
print(f"[*] Đang xử lý câu hỏi: {question}\n")

solution = orchestrator.solve(question)

print("=" * 60)
print("KẾT QUẢ XỬ LÝ:")
print("=" * 60)
print("Loại bài toán :", "Định tính (Giải thích hiện tượng)" if solution.is_qualitative else "Định lượng (Tính toán)")
print("Trạng thái    :", "✅ Thành công (Đã kiểm chứng)" if solution.is_verified else "❌ Chưa xác thực")
print("Số lần thử    :", solution.num_attempts)

if not solution.is_verified and solution.error_message:
    print("\n[!] LÝ DO LỖI / CHƯA XÁC THỰC:")
    print(f"    {solution.error_message}")

if solution.is_qualitative and solution.qualitative_output:
    print(f"\n[*] Nguyên lý áp dụng: {', '.join(solution.principles_applied)}")
    print("\n[*] Chuỗi giải thích nhân quả từng bước:")
    for step in solution.qualitative_output.causal_chain:
        print(f"  Bước {step.step_number}: {step.state_or_action}")
        print(f"    -> Cơ chế: {step.physical_mechanism}")
    print(f"\n[*] Kết luận: {solution.final_explanation}")
elif not solution.is_qualitative:
    print(f"\n[*] Đáp án: {solution.answer_value} {solution.answer_unit}")