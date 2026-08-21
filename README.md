# Neuro-Symbolic Physics Reasoning Engine for Small LLMs

A tool-augmented neuro-symbolic system that enables small LLMs (1B–8B parameters) to solve middle-school physics word problems with high reliability.

Instead of allowing the LLM to directly generate the final numerical calculation or equation derivations, this system combines:
- Natural Language Understanding (LLM)
- Physics Knowledge Base (YAML)
- Equation Retrieval & Matching
- Symbolic Equation Solving (SymPy)
- Unit Conversion & Dimensional Analysis (Pint)
- Multi-stage Verification Engine (7 verification checks)
- Verify-Repair Loop with Structured Feedback
- Benchmarking & Evaluation against Raw LLM and Calculator baselines

## Installation

Using `uv` (recommended):

```bash
uv venv
uv pip install -e ".[dev]"
```

## CLI Usage

### 1. Chế độ tương tác trực tiếp (Interactive Mode - Nhập câu hỏi liên tục):
```bash
uv run physics-engine interactive
# hoặc
uv run physics-engine solve -i
```
Trong chế độ này, bạn có thể gõ câu hỏi liên tục vào terminal, gõ `verbose` để bật/tắt chi tiết các bước giải, hoặc gõ `exit` / `quit` để thoát.

### 2. Giải từng câu hỏi trực tiếp trên dòng lệnh:
```bash
uv run physics-engine solve "Một vật có khối lượng 2 kg chịu tác dụng của lực 10 N. Tính gia tốc của vật."
```
Nếu chạy `uv run physics-engine solve` mà không truyền câu hỏi, chương trình sẽ tự động nhắc bạn nhập câu hỏi vào terminal:
```bash
uv run physics-engine solve
```

### 3. Xuất kết quả dưới định dạng JSON:
```bash
uv run physics-engine solve "A 2 kg object experiences a force of 10 N. Find its acceleration." --json
```

Validate the Physics Knowledge Base:
```bash
uv run physics-engine validate-kb
```

Run Benchmark Evaluation:
```bash
uv run physics-engine evaluate --dataset data/problems/splits/test.jsonl --experiment configs/experiments/full_system.yaml
```

## Running Tests

```bash
uv run pytest tests/ -v
```
