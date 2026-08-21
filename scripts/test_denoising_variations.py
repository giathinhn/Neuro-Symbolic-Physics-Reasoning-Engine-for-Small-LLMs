import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("src"))
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

variations = [
    {
        "name": "vd1 (Short with +)",
        "question": "Trộn 200g nước ở 80°C + 300g nước ở 20°C. Hãy cân bằng nhiệt",
    },
    {
        "name": "vd2 (Standard phrasing)",
        "question": "Trộn 200g nước ở 80°C với 300g nước ở 20°C. Nhiệt độ của nước khi cân bằng nhiệt là bao nhiêu?",
    },
    {
        "name": "vd3 (With boundary condition noise)",
        "question": "Trộn 200g nước ở 80°C với 300g nước ở 20°C. Bỏ qua sự mất mát nhiệt ra môi trường. Nhiệt độ của nước khi cân bằng nhiệt là bao nhiêu?",
    },
    {
        "name": "vd4 (Heavy noise: container & surrounding loss)",
        "question": "Trộn 200g nước ở 80°C với 300g nước ở 20°C. Bỏ qua sự truyền nhiệt cho bình chứa và môi trường xung quanh. Hãy tính nhiệt độ của nước khi có cân bằng nhiệt.",
    },
    {
        "name": "vd5 (Mechanics noise: ignoring friction and pulley mass)",
        "question": "Một cần trục nâng một thùng hàng có khối lượng 500 kg lên cao 6 m trong thời gian 15 s. Bỏ qua mọi ma sát và khối lượng của dây nối. Tính công suất trung bình của cần trục. Lấy g = 10 m/s^2.",
    },
]

def main():
    print("=" * 70)
    print("TESTING NOISE REDUCTION & ROBUSTNESS ACROSS PROBLEM PHRASINGS")
    print("=" * 70)
    for idx, v in enumerate(variations, 1):
        print(f"\n[{idx}/{len(variations)}] {v['name']}")
        print(f"  Câu hỏi: {v['question']}")
        sol = orchestrator.solve(v["question"])
        status = "✅ THÀNH CÔNG" if sol.is_verified else "❌ THẤT BẠI"
        print(f"  Trạng thái: {status} (Lần thử: {sol.num_attempts})")
        print(f"  Đáp án    : {sol.answer_value} {sol.answer_unit}")
        if not sol.is_verified and sol.error_message:
            print(f"  [!] Lỗi   : {sol.error_message}")

if __name__ == "__main__":
    main()
