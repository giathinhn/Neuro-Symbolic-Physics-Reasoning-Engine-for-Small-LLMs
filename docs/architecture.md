# System Architecture: Neuro-Symbolic Physics Reasoning Engine

## 1. High-Level Architecture Overview

The Neuro-Symbolic Physics Reasoning Engine pairs small Large Language Models (1B–8B parameter models like Phi-3-mini or Qwen2.5-3B) with deterministic symbolic computation tools to reliably solve physics word problems.

```
                      +-----------------------------+
                      |     User Physics Problem    |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Pipeline Orchestrator   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |        Small LLM            |
                      |   (Natural Language NLU)    |
                      +--------------+--------------+
                                     |  JSON Schema
                                     v
                      +-----------------------------+
                      |      Output Parser &        |
                      |     Quantity Extractor      |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Physics Knowledge Base    |
                      |    & Equation Retriever     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Unit Engine (Pint)        |
                      |  (Normalize givens to SI)   |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Symbolic Solver (SymPy)   |
                      |   (Exact algebraic solving) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   Verification Pipeline     |
                      |   (7 deterministic checks)  |
                      +--------------+--------------+
                                     |
                         +-----------+-----------+
                         |                       |
                      [PASS]                  [FAIL]
                         |                       |
                         v                       v
               +------------------+    +-------------------+
               |  Final Verified  |    |  Structured Error |
               |     Solution     |    |    Feedback       |
               +------------------+    +---------+---------+
                                                 |
                                                 v
                                       +-------------------+
                                       |  LLM Repair Loop  |
                                       |  (Up to 3 retries)|
                                       +-------------------+
```

## 2. Core Subsystems

### 2.1 LLM Subsystem
- **Provider Abstraction (`LLMProvider`)**: Supports LiteLLM (Ollama local models, OpenAI, Anthropic, HuggingFace) and a deterministic `MockLLMProvider` for testing.
- **Output Parser (`OutputParser`)**: Employs multi-tier extraction strategies (direct JSON, markdown code fence, outermost bracket matching) to ensure parsing robustness even when small LLMs add conversational wrappers.

### 2.2 Symbolic Physics Engine
- **Knowledge Base (`KnowledgeBase`)**: Memory-resident indexed database loaded from human-readable YAML equation files across kinematics, Newton's laws, work-energy, momentum, and density/pressure.
- **Equation Retriever (`EquationRetriever`)**: Ranks equations based on variable set overlap and matches equations using SymPy algebraic equivalence `simplify(lhs1 - rhs1 - (lhs2 - rhs2)) == 0`.
- **Unit Engine (`UnitEngine`)**: Wraps Pint to perform unit parsing, dimensional analysis, metric conversions, and standardization to SI base units.
- **Symbolic Solver (`SymbolicSolver`)**: Safe SymPy-based solver with algebraic rearrangement, variable elimination in multi-equation systems, and back-substitution validation.

### 2.3 Verification Pipeline
Executes 7 deterministic verification checks:
1. **Equation Validity Check**: Verifies syntax and existence in standard physics.
2. **Dimensional Analysis Check**: Verifies LHS and RHS dimensional consistency $[LHS] == [RHS]$.
3. **Unit Consistency Check**: Checks unit compatibility and target unit matching.
4. **Arithmetic Check**: Confirms solver produced real, finite numeric solutions.
5. **Physical Bounds Check**: Checks physical constraints (e.g. $m > 0$, $v < c$, $t \ge 0$).
6. **Substitution Check**: Substitutes calculated answers back into original equations and evaluates residual.
7. **System Consistency Check**: Detects contradictory multi-equation systems.

### 2.4 Verify-Repair Loop
When any verification check produces an `ERROR` or `FATAL` severity, the pipeline generates a structured error report containing actionable hints and initiates a repair turn with the LLM (up to `max_retries`).
