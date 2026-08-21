import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath("src"))
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

test_cases = [
    {
        "id": "Q1_FreeFall",
        "question": "Một vật rơi tự do từ độ cao 45 m xuống đất. Lấy g = 10 m/s^2. Tính thời gian rơi của vật.",
        "expected": "3.0 s",
        "expected_val": 3.0,
    },
    {
        "id": "Q2_UniformMotion",
        "question": "Một người đi xe đạp chuyển động đều với vận tốc 15 km/h trong thời gian 2 giờ. Tính quãng đường người đó đi được.",
        "expected": "30.0 km",
        "expected_val": 30.0,
    },
    {
        "id": "Q3_SolidPressure",
        "question": "Một cái ghế có khối lượng 50 kg có 4 chân, diện tích tiếp xúc của mỗi chân với mặt sàn là 0,002 m^2. Tổng diện tích tiếp xúc là 0,008 m^2. Lấy g = 10 m/s^2. Tính áp suất của ghế tác dụng lên mặt sàn.",
        "expected": "62500.0 Pa",
        "expected_val": 62500.0,
    },
    {
        "id": "Q4_CranePower_Mass",
        "question": "Một cần trục nâng một thùng hàng có khối lượng 500 kg lên cao 6 m trong thời gian 15 s. Công suất trung bình của cần trục là bao nhiêu? (Lấy g = 10 m/s^2).",
        "expected": "2000.0 W",
        "expected_val": 2000.0,
    },
    {
        "id": "Q5_CranePower_Force",
        "question": "Một cần cẩu nâng một vật có trọng lượng 2000 N lên cao 5 m trong thời gian 10 giây. Tính công suất của cần cẩu.",
        "expected": "1000.0 W",
        "expected_val": 1000.0,
    },
    {
        "id": "Q6_ParallelCircuit",
        "question": "Hai điện trở R1 = 20 ohm và R2 = 30 ohm mắc song song với nhau. Tính điện trở tương đương của đoạn mạch.",
        "expected": "12.0 ohm",
        "expected_val": 12.0,
    },
    {
        "id": "Q7_OhmsLaw",
        "question": "Đặt một hiệu điện thế U = 24 V vào hai đầu điện trở R = 12 ohm. Tính cường độ dòng điện chạy qua điện trở.",
        "expected": "2.0 A",
        "expected_val": 2.0,
    },
    {
        "id": "Q8_ElectricPower",
        "question": "Một bóng đèn có hiệu điện thế định mức U = 220 V và cường độ dòng điện định mức I = 0,5 A. Tính công suất điện định mức của bóng đèn.",
        "expected": "110.0 W",
        "expected_val": 110.0,
    },
    {
        "id": "Q9_HeatBalance",
        "question": "Trộn 200 g nước ở 80 °C với 300 g nước ở 20 °C. Bỏ qua sự mất mát nhiệt ra môi trường. Nhiệt độ của nước khi cân bằng nhiệt là bao nhiêu?",
        "expected": "44.0 °C",
        "expected_val": 44.0,
    },
    {
        "id": "Q10_Archimedes",
        "question": "Một vật có thể tích 0,002 m^3 được nhúng chìm hoàn toàn trong nước. Biết trọng lượng riêng của nước là 10000 N/m^3. Tính lực đẩy Ác-si-mét tác dụng lên vật.",
        "expected": "20.0 N",
        "expected_val": 20.0,
    },
]

def main():
    print("=" * 70)
    print("BENCHMARKING 10 EXTENDED THCS QUANTITATIVE PHYSICS PROBLEMS")
    print("=" * 70)
    
    orchestrator = PipelineOrchestrator()
    passed_count = 0
    
    for idx, tc in enumerate(test_cases, 1):
        print(f"\n[{idx}/{len(test_cases)}] Đang xử lý: {tc['id']}")
        print(f"    Câu hỏi: {tc['question']}")
        
        t0 = time.time()
        res = orchestrator.solve(tc["question"])
        elapsed = time.time() - t0
        
        if res.is_verified and res.answer_value is not None:
            val = float(res.answer_value)
            exp_val = tc["expected_val"]
            if abs(val - exp_val) <= 0.05 * abs(exp_val) or abs(val - exp_val) <= 0.1:
                print(f"    Trạng thái: ✅ THÀNH CÔNG ({elapsed:.2f}s, {res.num_attempts} lần thử)")
                print(f"    Kết quả tìm được : {res.answer_value} {res.answer_unit}")
                print(f"    Kết quả kỳ vọng  : {tc['expected']}")
                passed_count += 1
            else:
                print(f"    Trạng thái: ⚠️ SAI SỐ SỐ HỌC ({elapsed:.2f}s, {res.num_attempts} lần thử)")
                print(f"    Kết quả tìm được : {res.answer_value} {res.answer_unit}")
                print(f"    Kết quả kỳ vọng  : {tc['expected']}")
        else:
            print(f"    Trạng thái: ❌ THẤT BẠI ({elapsed:.2f}s, {res.num_attempts} lần thử)")
            print(f"    Kết quả tìm được : {res.answer_value} {res.answer_unit}")
            print(f"    Kết quả kỳ vọng  : {tc['expected']}")
            if res.error_message:
                print(f"    [!] Lỗi: {res.error_message}")
                
    print("\n" + "=" * 70)
    print(f"KẾT QUẢ TỔNG HỢP: {passed_count}/{len(test_cases)} BÀI TOÁN ĐẠT CHUẨN XÁC THỰC ({passed_count/len(test_cases)*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
