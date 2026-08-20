import json
import urllib.request
import time

print("Sending request to Ollama...")
t0 = time.time()
req = urllib.request.Request(
    "http://localhost:11434/api/chat",
    data=json.dumps({
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "Tại sao cánh quạt bị bám bụi sau một thời gian sử dụng? Trả lời ngắn gọn 2 câu."}],
        "stream": False
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
        print(f"Elapsed: {time.time() - t0:.2f}s")
        print("Response:", data["message"]["content"])
except Exception as e:
    print("Error:", e)
