"""Demonstration script executing Qualitative Neuro-Symbolic Physics Reasoning on Middle School (THCS) problems."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from physics_reasoning.core.config import PipelineConfig
from physics_reasoning.llm.provider import MockLLMProvider
from physics_reasoning.physics.qualitative_kb import QualitativeKnowledgeBase
from physics_reasoning.pipeline.orchestrator import PipelineOrchestrator


def main():
    print("=" * 70)
    print("  DEMO: NEURO-SYMBOLIC QUALITATIVE REASONING FOR THCS PHYSICS")
    print("=" * 70)

    kb = QualitativeKnowledgeBase("data/knowledge/qualitative")
    kb.load()
    print(f"[*] Loaded {len(kb.principles)} qualitative physical principles.")
    print(f"[*] Loaded {len(kb.misconceptions)} anti-misconception patterns.\n")

    # Sample qualitative problems and ideal structured outputs
    test_cases = [
        {
            "problem": "Tại sao khi xe buýt đang chạy mà phanh gấp thì hành khách ngồi trên xe lại bị ngã chúi về phía trước?",
            "topic": "Quán tính (Vật lý 8)",
            "output": {
                "problem_understanding": "Giải thích hiện tượng người bị chúi về phía trước khi xe phanh gấp",
                "observed_phenomenon": "Hành khách ngã chúi về phía trước khi xe hãm phanh đột ngột",
                "core_principles": ["inertia_law"],
                "causal_chain": [
                    {
                        "step_number": 1,
                        "state_or_action": "Xe và toàn bộ cơ thể hành khách đang chuyển động về phía trước với cùng vận tốc v.",
                        "physical_mechanism": "Khi xe phanh gấp, lực ma sát làm xe và phần thân dưới (tiếp xúc với ghế/sàn) giảm nhanh vận tốc và dừng lại.",
                        "governing_principle": "inertia_law",
                    },
                    {
                        "step_number": 2,
                        "state_or_action": "Phần thân trên của hành khách chưa chịu lực hãm trực tiếp.",
                        "physical_mechanism": "Do có quán tính, phần thân trên tiếp tục duy trì vận tốc v cũ hướng về phía trước.",
                        "governing_principle": "inertia_law",
                    },
                    {
                        "step_number": 3,
                        "state_or_action": "Sự chênh lệch trạng thái chuyển động giữa hai phần cơ thể.",
                        "physical_mechanism": "Phần thân trên di chuyển nhanh hơn phần thân dưới làm hành khách có xu hướng bị ngã chúi về phía trước.",
                        "governing_principle": "inertia_law",
                    },
                ],
                "conclusion": "Khi xe phanh gấp, do có quán tính nên phần thân trên của hành khách tiếp tục duy trì vận tốc cũ trong khi phần dưới đã dừng lại cùng xe, khiến người bị ngã chúi về phía trước.",
                "scientific_keywords": ["quán tính", "duy trì vận tốc", "phanh gấp"],
            },
        },
        {
            "problem": "Tại sao đầu đinh và mũi kim lại được làm nhọn, trong khi chân ghế, chân bàn và móng nhà lại làm to bản?",
            "topic": "Áp suất chất rắn p = F / S (Vật lý 8)",
            "output": {
                "problem_understanding": "Giải thích cấu tạo đầu đinh nhọn và móng nhà to bản dựa trên áp suất",
                "observed_phenomenon": "Đinh nhọn dễ đóng vào gỗ, móng nhà to bản tránh bị lún đất",
                "core_principles": ["solid_pressure_area"],
                "causal_chain": [
                    {
                        "step_number": 1,
                        "state_or_action": "Tác dụng của lực ép (áp lực F) lên bề mặt tiếp xúc.",
                        "physical_mechanism": "Áp suất chất rắn được xác định bằng công thức p = F / S.",
                        "governing_principle": "solid_pressure_area",
                    },
                    {
                        "step_number": 2,
                        "state_or_action": "Trường hợp đầu đinh, mũi kim",
                        "physical_mechanism": "Đầu đinh và mũi kim được mài nhọn làm giảm diện tích tiếp xúc S, với cùng áp lực F sẽ tạo ra áp suất p rất lớn giúp đinh/kim dễ dàng xuyên thủng bề mặt vật.",
                        "governing_principle": "solid_pressure_area",
                    },
                    {
                        "step_number": 3,
                        "state_or_action": "Trường hợp chân ghế, móng nhà",
                        "physical_mechanism": "Chân ghế và móng nhà làm to bản để tăng diện tích tiếp xúc S, với cùng trọng lượng của công trình sẽ làm giảm áp suất p tác dụng lên nền đất, chống lún nứt.",
                        "governing_principle": "solid_pressure_area",
                    },
                ],
                "conclusion": "Theo công thức p = F/S: đầu đinh nhọn làm giảm diện tích tiếp xúc S để tăng áp suất p giúp dễ đâm xuyên; móng nhà to bản làm tăng diện tích S để giảm áp suất p giúp tránh lún công trình.",
                "scientific_keywords": ["áp suất", "áp lực", "diện tích tiếp xúc"],
            },
        },
        {
            "problem": "Tại sao khi rót nước sôi vào cốc thủy tinh dày thì cốc dễ bị nứt vỡ hơn so với cốc mỏng?",
            "topic": "Sự dãn nở vì nhiệt & Dẫn nhiệt (Vật lý 6/8)",
            "output": {
                "problem_understanding": "Giải thích sự nứt vỡ của cốc thủy tinh dày khi rót nước sôi",
                "observed_phenomenon": "Cốc thủy tinh dày dễ vỡ hơn cốc mỏng khi gặp nước sôi đột ngột",
                "core_principles": ["thermal_expansion_unequal"],
                "causal_chain": [
                    {
                        "step_number": 1,
                        "state_or_action": "Rót nước sôi vào lòng cốc thủy tinh.",
                        "physical_mechanism": "Lớp thủy tinh bên trong tiếp xúc trực tiếp với nước sôi nóng lên ngay và dãn nở vì nhiệt.",
                        "governing_principle": "thermal_expansion_unequal",
                    },
                    {
                        "step_number": 2,
                        "state_or_action": "Đặc tính truyền nhiệt của thủy tinh.",
                        "physical_mechanism": "Thủy tinh là chất dẫn nhiệt kém; ở cốc dày, nhiệt chưa kịp truyền tới lớp bên ngoài nên lớp ngoài chưa kịp nóng và chưa dãn nở.",
                        "governing_principle": "thermal_expansion_unequal",
                    },
                    {
                        "step_number": 3,
                        "state_or_action": "Sự dãn nở không đồng đều giữa hai mặt thành cốc.",
                        "physical_mechanism": "Lớp trong dãn nở nhưng bị lớp ngoài ngăn cản sinh ra ứng suất nhiệt và nội lực rất lớn làm nứt vỡ cốc. Cốc mỏng truyền nhiệt nhanh nên hai mặt dãn nở đồng đều không bị nứt.",
                        "governing_principle": "thermal_expansion_unequal",
                    },
                ],
                "conclusion": "Do thủy tinh dẫn nhiệt kém, ở cốc dày lớp bên trong nở ra trước khi lớp ngoài chưa kịp nở, sự dãn nở không đều gây ra lực lớn làm nứt vỡ cốc.",
                "scientific_keywords": ["dãn nở vì nhiệt", "dẫn nhiệt kém", "nứt vỡ"],
            },
        },
    ]

    for tc in test_cases:
        prob = tc["problem"]
        print(f"----------------------------------------------------------------------")
        print(f"[*] CHỦ ĐỀ: {tc['topic']}")
        print(f"[*] CÂU HỎI: {prob}\n")

        mock_llm = MockLLMProvider(responses=[json.dumps(tc["output"])])
        orchestrator = PipelineOrchestrator(
            config=PipelineConfig(max_retries=3),
            llm_provider=mock_llm,
            qualitative_knowledge_base=kb,
        )

        solution = orchestrator.solve(prob)

        print(f"[*] KẾT QUẢ XÁC THỰC: {'[HỢP LỆ / ĐÃ KIỂM CHỨNG]' if solution.is_verified else '[KHÔNG HỢP LỆ]'}")
        print(f"[*] NGUYÊN LÝ ÁP DỤNG: {', '.join(solution.principles_applied)}")
        print(f"[*] CHUỖI NHÂN QUẢ CHI TIẾT:")
        if solution.qualitative_output:
            for s in solution.qualitative_output.causal_chain:
                print(f"    - Bước {s.step_number}: {s.state_or_action}")
                print(f"      + Cơ chế vật lý: {s.physical_mechanism}")

        print(f"\n[*] LỜI KẾT LUẬN GIẢI THÍCH:")
        print(f"    \"{solution.final_explanation}\"\n")


if __name__ == "__main__":
    main()
