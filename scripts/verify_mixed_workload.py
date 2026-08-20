"""Verification script testing mixed quantitative and qualitative physics problems."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.core.enums import ProblemType
from physics_reasoning.llm.provider import MockLLMProvider
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase
from physics_reasoning.pipeline.classifier import ProblemClassifier
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


def main():
    print("=" * 75)
    print("  VERIFYING MIXED WORKLOAD: QUANTITATIVE & QUALITATIVE PROBLEMS")
    print("=" * 75)

    quant_kb = KnowledgeBase("data/knowledge")
    quant_kb.load()
    qual_kb = QualitativeKnowledgeBase("data/knowledge/qualitative")
    qual_kb.load()

    problems = [
        # --- QUANTITATIVE PROBLEMS ---
        {
            "id": "quant_001",
            "type": ProblemType.QUANTITATIVE,
            "text": "A 5 kg block is acted upon by a net force of 25 N. Calculate its acceleration.",
            "response": json.dumps(
                {
                    "problem_understanding": "Find acceleration given mass and force",
                    "quantities": [
                        {"name": "mass", "symbol": "m", "value": 5.0, "unit": "kg", "role": "given"},
                        {"name": "force", "symbol": "F", "value": 25.0, "unit": "N", "role": "given"},
                        {"name": "acceleration", "symbol": "a", "value": None, "unit": "m/s**2", "role": "target"},
                    ],
                    "equations": [
                        {"equation_id": "newton2", "expression": "F = m * a", "justification": "Newton's second law"}
                    ],
                    "target_variable": "a",
                }
            ),
            "expected_ans": 5.0,
            "expected_unit": "m / s ** 2",
        },
        {
            "id": "quant_002",
            "type": ProblemType.QUANTITATIVE,
            "text": "An object moves at a constant speed of 72 km/h for 10 seconds. Find the distance traveled.",
            "response": json.dumps(
                {
                    "problem_understanding": "Find distance given speed in km/h and time",
                    "quantities": [
                        {"name": "velocity", "symbol": "v", "value": 72.0, "unit": "km/h", "role": "given"},
                        {"name": "time", "symbol": "t", "value": 10.0, "unit": "s", "role": "given"},
                        {"name": "distance", "symbol": "d", "value": None, "unit": "m", "role": "target"},
                    ],
                    "equations": [
                        {"equation_id": "kin_vel_def", "expression": "v = d / t", "justification": "Velocity definition"}
                    ],
                    "target_variable": "d",
                }
            ),
            "expected_ans": 200.0,
            "expected_unit": "m",
        },
        {
            "id": "quant_003",
            "type": ProblemType.QUANTITATIVE,
            "text": "Một vật có khối lượng 2 kg chuyển động với vận tốc 6 m/s. Tính động năng của vật bằng bao nhiêu Jun?",
            "response": json.dumps(
                {
                    "problem_understanding": "Tính động năng từ khối lượng và vận tốc",
                    "quantities": [
                        {"name": "mass", "symbol": "m", "value": 2.0, "unit": "kg", "role": "given"},
                        {"name": "velocity", "symbol": "v", "value": 6.0, "unit": "m/s", "role": "given"},
                        {"name": "kinetic_energy", "symbol": "KE", "value": None, "unit": "J", "role": "target"},
                    ],
                    "equations": [
                        {"equation_id": "ke_def", "expression": "KE = (1/2) * m * v^2", "justification": "Định nghĩa động năng"}
                    ],
                    "target_variable": "KE",
                }
            ),
            "expected_ans": 36.0,
            "expected_unit": "J",
        },

        # --- QUALITATIVE PHENOMENON EXPLANATIONS ---
        {
            "id": "qual_001",
            "type": ProblemType.QUALITATIVE,
            "text": "Tại sao khi xe buýt phanh gấp thì hành khách ngồi trên xe lại bị ngã chúi về phía trước?",
            "response": json.dumps(
                {
                    "problem_understanding": "Giải thích hiện tượng ngã chúi khi phanh gấp",
                    "observed_phenomenon": "Người ngã chúi về trước",
                    "core_principles": ["inertia_law"],
                    "causal_chain": [
                        {
                            "step_number": 1,
                            "state_or_action": "Xe và toàn bộ người đang cùng chuyển động về trước với vận tốc v.",
                            "physical_mechanism": "Khi phanh gấp, lực ma sát hãm xe và chân người dừng lại.",
                            "governing_principle": "inertia_law",
                        },
                        {
                            "step_number": 2,
                            "state_or_action": "Phần thân trên của người",
                            "physical_mechanism": "Do có quán tính nên phần thân trên tiếp tục duy trì vận tốc cũ chuyển động về phía trước.",
                            "governing_principle": "inertia_law",
                        },
                    ],
                    "conclusion": "Khi xe phanh gấp, do có quán tính nên thân trên tiếp tục duy trì vận tốc cũ khiến người ngã chúi về trước.",
                    "scientific_keywords": ["quán tính", "duy trì vận tốc"],
                }
            ),
            "expected_principle": "inertia_law",
        },
        {
            "id": "qual_002",
            "type": ProblemType.QUALITATIVE,
            "text": "Tại sao một khối sắt đặc bị chìm khi thả vào nước nhưng con tàu thép khổng lồ lại nổi trên biển?",
            "response": json.dumps(
                {
                    "problem_understanding": "Giải thích điều kiện nổi của tàu thủy bằng thép",
                    "observed_phenomenon": "Bi sắt chìm nhưng tàu thép nổi",
                    "core_principles": ["archimedes_buoyancy_floating"],
                    "causal_chain": [
                        {
                            "step_number": 1,
                            "state_or_action": "Cấu tạo khoang rỗng của tàu thủy",
                            "physical_mechanism": "Tàu thủy được thiết kế có các khoang rỗng lớn làm tăng thể tích chiếm chỗ V của tàu trong nước.",
                            "governing_principle": "archimedes_buoyancy_floating",
                        },
                        {
                            "step_number": 2,
                            "state_or_action": "So sánh trọng lượng riêng trung bình",
                            "physical_mechanism": "Thể tích lớn làm trọng lượng riêng trung bình của tàu nhỏ hơn trọng lượng riêng của nước, do đó lực đẩy Ác-si-mét nâng tàu nổi trên mặt nước.",
                            "governing_principle": "archimedes_buoyancy_floating",
                        },
                    ],
                    "conclusion": "Tàu thủy có các khoang rỗng lớn làm thể tích chiếm chỗ rất lớn, khiến trọng lượng riêng trung bình của tàu nhỏ hơn nước nên tàu nổi.",
                    "scientific_keywords": ["lực đẩy ác-si-mét", "trọng lượng riêng", "khoang rỗng"],
                }
            ),
            "expected_principle": "archimedes_buoyancy_floating",
        },
    ]

    all_passed = True
    for item in problems:
        p_id = item["id"]
        p_text = item["text"]
        expected_type = item["type"]

        # 1. Test Classifier
        classified_type = ProblemClassifier.classify(p_text)
        type_match = classified_type == expected_type

        # 2. Test Execution
        mock_llm = MockLLMProvider(responses=[item["response"]])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=2),
            llm_provider=mock_llm,
            knowledge_base=quant_kb,
            qualitative_knowledge_base=qual_kb,
        )
        sol = orchestrator.solve(p_text)

        status_ok = sol.is_verified
        if expected_type == ProblemType.QUANTITATIVE:
            val_ok = sol.answer_value is not None and abs(sol.answer_value - item["expected_ans"]) < 0.01
            unit_ok = sol.answer_unit is not None
            details = f"Ans: {sol.answer_value} {sol.answer_unit} (Expected: {item['expected_ans']} {item['expected_unit']})"
            test_success = type_match and status_ok and val_ok
        else:
            principle_ok = item["expected_principle"] in sol.principles_applied
            details = f"Principles: {sol.principles_applied} | Causal Steps: {len(sol.qualitative_output.causal_chain) if sol.qualitative_output else 0}"
            test_success = type_match and status_ok and principle_ok

        mark = "[PASS]" if test_success else "[FAIL]"
        if not test_success:
            all_passed = False

        print(f"{mark} [{p_id}] Type: {classified_type.upper():12s} | Verified: {str(sol.is_verified):5s} | {details}")

    print("-" * 75)
    if all_passed:
        print("[*] RESULT: ALL MIXED QUANTITATIVE & QUALITATIVE WORKLOADS PASSED PERFECTLY!\n")
    else:
        print("[!] RESULT: SOME WORKLOADS FAILED.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
