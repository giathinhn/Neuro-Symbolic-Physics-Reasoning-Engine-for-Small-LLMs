import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("src"))
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

qualitative_questions = [
    {
        "domain": "Điện học - Tĩnh điện",
        "question": "Tại sao cánh quạt bị bám bụi sau một thời gian sử dụng?",
        "expected_principle": "electrostatic_charging_friction",
    },
    {
        "domain": "Cơ học - Quán tính",
        "question": "Tại sao khi xe phanh gấp, hành khách ngồi trên xe bị ngã chúi về phía trước?",
        "expected_principle": "inertia_law",
    },
    {
        "domain": "Nhiệt học - Chuyển thể",
        "question": "Tại sao mặt ngoài của cốc nước đá sau một lúc lại có các giọt nước đọng lại?",
        "expected_principle": "condensation_dew_formation",
    },
    {
        "domain": "Quang học - Khúc xạ",
        "question": "Tại sao cắm chiếc đũa thẳng vào cốc nước nhìn thấy đũa như bị gãy khúc ở mặt nước?",
        "expected_principle": "light_refraction_phenomena",
    },
    {
        "domain": "Cơ học - Áp suất khí quyển",
        "question": "Tại sao khi dùng ống hút hút bớt không khí trong hộp sữa, hộp sữa lại bị bẹp dúm lại?",
        "expected_principle": "atmospheric_pressure_effects",
    },
    {
        "domain": "Cơ học - Bình thông nhau",
        "question": "Tại sao vòi của ấm đun nước luôn được thiết kế cao ngang bằng miệng ấm?",
        "expected_principle": "communicating_vessels_principle",
    },
]


def main():
    print("=" * 75)
    print("BENCHMARK: QUALITATIVE PHYSICS PHENOMENON EXPLANATION SUITE")
    print("=" * 75)

    passed_count = 0
    for idx, item in enumerate(qualitative_questions, 1):
        print(f"\n[{idx}/{len(qualitative_questions)}] Chủ đề: {item['domain']}")
        print(f"  Câu hỏi: {item['question']}")

        sol = orchestrator.solve(item["question"])
        status = "✅ THÀNH CÔNG (ĐÃ XÁC THỰC)" if sol.is_verified else "❌ CHƯA XÁC THỰC"
        print(f"  Trạng thái  : {status}")
        print(f"  Số lần thử  : {sol.num_attempts}")
        print(f"  Nguyên lý   : {sol.principles_applied}")
        print(f"  Cơ chế / Kết luận: {sol.final_explanation}")
        if sol.is_verified:
            passed_count += 1
        elif sol.error_message:
            print(f"  [!] Lỗi     : {sol.error_message}")

    print("\n" + "=" * 75)
    print(f"TỔNG KẾT: {passed_count}/{len(qualitative_questions)} câu định tính đạt chuẩn kiểm chứng vật lý.")
    print("=" * 75)


if __name__ == "__main__":
    main()
