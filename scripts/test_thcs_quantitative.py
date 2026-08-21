"""Test THCS quantitative physics problems."""

import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("src"))

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

PROBLEMS = [
    {
        "id": "Q1_FreeFall",
        "text": "Một vật rơi tự do từ độ cao 45 m xuống đất. Lấy g = 10 m/s^2. Tính thời gian rơi của vật.",
        "expected": "3.0 s",
        "expected_val": 3.0,
    },
    {
        "id": "Q2_UniformMotion",
        "text": "Một người đi xe đạp chuyển động đều với vận tốc 15 km/h trong thời gian 2 giờ. Tính quãng đường người đó đi được.",
        "expected": "30.0 km (hoặc 30000 m)",
        "expected_val": 30000.0,
    },
    {
        "id": "Q3_SolidPressure",
        "text": "Một cái ghế có khối lượng 50 kg có 4 chân, diện tích tiếp xúc của mỗi chân với mặt sàn là 0,002 m^2. Tổng diện tích tiếp xúc là 0,008 m^2. Lấy g = 10 m/s^2. Tính áp suất của ghế tác dụng lên mặt sàn.",
        "expected": "62500.0 Pa",
        "expected_val": 62500.0,
    },
    {
        "id": "Q4_MechanicalPower",
        "text": "Một cần cẩu nâng một vật có trọng lượng 2000 N lên cao 5 m trong thời gian 10 giây. Tính công suất của cần cẩu.",
        "expected": "1000.0 W",
        "expected_val": 1000.0,
    },
    {
        "id": "Q5_ParallelCircuit",
        "text": "Hai điện trở R1 = 20 ohm và R2 = 30 ohm mắc song song với nhau. Tính điện trở tương đương của đoạn mạch.",
        "expected": "12.0 ohm",
        "expected_val": 12.0,
    },
    {
        "id": "Q6_OhmsLaw",
        "text": "Đặt một hiệu điện thế U = 24 V vào hai đầu điện trở R = 12 ohm. Tính cường độ dòng điện chạy qua điện trở.",
        "expected": "2.0 A",
        "expected_val": 2.0,
    },
    {
        "id": "Q7_HeatBalance",
        "text": "Trộn 200 g nước ở 80 °C với 300 g nước ở 20 °C. Bỏ qua sự mất mát nhiệt ra môi trường. Nhiệt độ của nước khi cân bằng nhiệt là bao nhiêu?",
        "expected": "44.0 °C",
        "expected_val": 44.0,
    },
    {
        "id": "Q8_Archimedes",
        "text": "Một vật có thể tích 0,002 m^3 được nhúng chìm hoàn toàn trong nước. Biết trọng lượng riêng của nước là 10000 N/m^3. Tính lực đẩy Ác-si-mét tác dụng lên vật.",
        "expected": "20.0 N",
        "expected_val": 20.0,
    },
]

def main():
    orchestrator = PipelineOrchestrator()
    print("=" * 70)
    print("BENCHMARKING THCS QUANTITATIVE PHYSICS PROBLEMS")
    print("=" * 70)
    
    passed = 0
    total = len(PROBLEMS)

    for i, p in enumerate(PROBLEMS, 1):
        print(f"\n[{i}/{total}] Đang xử lý: {p['id']}", flush=True)
        print(f"    Câu hỏi: {p['text']}", flush=True)
        start = time.time()
        try:
            sol = orchestrator.solve(p["text"])
            elapsed = time.time() - start
            status = "✅ THÀNH CÔNG" if sol.is_verified else "❌ THẤT BẠI"
            print(f"    Trạng thái: {status} ({elapsed:.2f}s, {sol.num_attempts} lần thử)", flush=True)
            print(f"    Kết quả tìm được : {sol.answer_value} {sol.answer_unit}", flush=True)
            print(f"    Kết quả kỳ vọng  : {p['expected']}", flush=True)
            if sol.is_verified:
                passed += 1
            else:
                if sol.error_message:
                    print(f"    [!] Lỗi: {sol.error_message}", flush=True)
        except Exception as e:
            print(f"    [!] EXCEPTION: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"KẾT QUẢ TỔNG HỢP: {passed}/{total} BÀI TOÁN ĐẠT CHUẨN XÁC THỰC ({(passed/total)*100:.1f}%)", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
