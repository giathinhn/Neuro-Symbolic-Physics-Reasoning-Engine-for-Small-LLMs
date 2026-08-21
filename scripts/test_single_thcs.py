import sys
import os
import time

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("src"))

from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator()

test_problems = [
    ("Q1", "Một vật rơi tự do từ độ cao 45 m xuống đất. Lấy g = 10 m/s^2. Tính thời gian rơi của vật."),
    ("Q2", "Một người đi xe đạp chuyển động đều với vận tốc 15 km/h trong thời gian 2 giờ. Tính quãng đường người đó đi được."),
    ("Q3", "Một cái ghế có khối lượng 50 kg có 4 chân, diện tích tiếp xúc của mỗi chân với mặt sàn là 0,002 m^2. Tổng diện tích tiếp xúc là 0,008 m^2. Lấy g = 10 m/s^2. Tính áp suất của ghế tác dụng lên mặt sàn."),
    ("Q4", "Một cần cẩu nâng một vật có trọng lượng 2000 N lên cao 5 m trong thời gian 10 giây. Tính công suất của cần cẩu."),
    ("Q5", "Hai điện trở R1 = 20 ohm và R2 = 30 ohm mắc song song với nhau. Tính điện trở tương đương của đoạn mạch."),
    ("Q6", "Đặt một hiệu điện thế U = 24 V vào hai đầu điện trở R = 12 ohm. Tính cường độ dòng điện chạy qua điện trở."),
    ("Q7", "Trộn 200 g nước ở 80 °C với 300 g nước ở 20 °C. Bỏ qua sự mất mát nhiệt ra môi trường. Nhiệt độ của nước khi cân bằng nhiệt là bao nhiêu?"),
    ("Q8", "Một vật có thể tích 0,002 m^3 được nhúng chìm hoàn toàn trong nước. Biết trọng lượng riêng của nước là 10000 N/m^3. Tính lực đẩy Ác-si-mét tác dụng lên vật."),
]

for q_id, text in test_problems:
    print(f"\n====================\n[*] SOLVING {q_id}: {text}", flush=True)
    t0 = time.time()
    try:
        sol = orchestrator.solve(text)
        print(f"Verified: {sol.is_verified}", flush=True)
        print(f"Answer: {sol.answer_value} {sol.answer_unit}", flush=True)
        print(f"Attempts: {sol.attempts_used}, Time: {time.time()-t0:.2f}s", flush=True)
        if not sol.is_verified:
            print(f"Error: {sol.error_message}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
