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

Solve a physics problem:
```bash
uv run physics-engine solve "A 2 kg object experiences a force of 10 N. Find its acceleration."
```

Output in JSON format:
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
