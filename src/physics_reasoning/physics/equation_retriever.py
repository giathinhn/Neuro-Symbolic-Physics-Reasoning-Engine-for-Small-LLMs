"""Equation retrieval and semantic matching against knowledge base."""

from __future__ import annotations

from sympy import Eq, simplify

from physics_reasoning.core.models import Equation
from physics_reasoning.physics.knowledge_base import KnowledgeBase
from physics_reasoning.solver.expression_parser import parse_equation_string


class EquationRetriever:
    """Retrieve and match equations from the knowledge base."""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve_by_quantities(
        self,
        quantity_names: list[str],
        topic: str | None = None,
        top_k: int = 5,
    ) -> list[Equation]:
        """Retrieve top-k equations matching a set of quantities."""
        results = self.kb.search_by_quantities(quantity_names, topic=topic)
        return results[:top_k]

    def match_expression(self, expression_str: str) -> Equation | None:
        """Match a free-form equation string to a knowledge base equation."""
        if not expression_str or "=" not in expression_str:
            return None

        # 1. Exact string match
        clean = expression_str.replace(" ", "")
        for eq in self.kb.equations.values():
            if eq.expression.replace(" ", "") == clean:
                return eq

        # 2. SymPy algebraic equivalence match
        try:
            parsed_input = parse_equation_string(expression_str)
            input_diff = parsed_input.lhs - parsed_input.rhs

            for eq in self.kb.equations.values():
                try:
                    kb_parsed = parse_equation_string(eq.expression)
                    kb_diff = kb_parsed.lhs - kb_parsed.rhs

                    if simplify(input_diff - kb_diff) == 0 or simplify(input_diff + kb_diff) == 0:
                        return eq
                except Exception:
                    continue
        except Exception:
            return None

        return None

    def suggest_equations(
        self, given_quantities: list[str], target_quantity: str
    ) -> list[Equation]:
        """Find equations that connect the given quantities to the target quantity."""
        target_qty = self.kb.get_quantity_by_name(target_quantity) or self.kb.get_quantity_by_symbol(target_quantity)
        target_name = target_qty.name if target_qty else target_quantity

        given_canonical: set[str] = set()
        for g in given_quantities:
            qty = self.kb.get_quantity_by_name(g) or self.kb.get_quantity_by_symbol(g)
            given_canonical.add(qty.name if qty else g)

        matching_eqs: list[Equation] = []
        for eq in self.kb.equations.values():
            var_names = set(eq.variable_quantities.values())
            if target_name in var_names and len(var_names.intersection(given_canonical)) > 0:
                matching_eqs.append(eq)

        return matching_eqs

    def retrieve_for_problem(
        self,
        problem_text: str,
        given_symbols: list[str] | None = None,
        target_symbol: str | None = None,
    ) -> list[Equation]:
        """Retrieve relevant equations for a physics problem based on keywords and symbols."""
        p_lower = problem_text.lower()
        matched_ids: list[str] = []

        # Vietnamese physics domain keywords
        if any(w in p_lower for w in ["rơi tự do", "thả rơi", "độ cao", "free fall", "chạm đất"]):
            matched_ids.extend(["kin_free_fall", "force_gravity_mg", "weight_formula", "weight_P"])
        if any(w in p_lower for w in ["vận tốc", "quãng đường", "xe đạp", "chuyển động", "thời gian", "speed", "velocity", "gặp nhau", "đuổi theo"]):
            matched_ids.extend(["kin_uniform_motion_s", "kin_vel_def", "kin_eq1", "kin_eq2"])
        if any(w in p_lower for w in ["áp suất", "ghế", "tiếp xúc", "diện tích", "pressure", "chân ghế", "sàn", "đáy bình", "chất lỏng", "độ sâu"]):
            matched_ids.extend(["solid_pressure", "hydrostatic_pressure_depth", "hydrostatic_pressure_rho_g_h", "force_gravity_mg", "weight_formula", "weight_P"])
        if any(w in p_lower for w in ["cần cẩu", "cần trục", "công suất", "công cơ học", "nâng", "kéo", "power", "work", "thùng hàng"]):
            matched_ids.extend(["power_lifting_mght", "power_avg_mght", "power_force_height_time", "power_force_dist_time", "power_from_work_time", "work_lifting_mgh", "work_lifting_P_h", "work_A_def", "work_def", "force_gravity_mg", "weight_formula", "power_def"])
        if any(w in p_lower for w in ["song song", "nối tiếp", "điện trở", "hiệu điện thế", "mạch", "ohm", "cường độ", "resistor", "công suất điện", "điện năng", "tiêu thụ", "jun-len-xơ", "nhiệt lượng tỏa ra trên"]):
            matched_ids.extend(["parallel_resistors_two", "ohm_law", "series_resistors_two", "electric_power_ui", "electric_power_i2r", "electric_power_u2_div_r", "electric_energy_uit", "joule_lenz_heat"])
        if any(w in p_lower for w in ["trộn", "nhiệt độ", "cân bằng nhiệt", "nước ở", "tỏa", "thu", "heat", "temperature", "đun sôi", "nhiệt lượng cần", "nhiên liệu", "đốt cháy"]):
            matched_ids.extend(["thermal_equilibrium", "thermal_equilibrium_general", "heat_absorption_delta_t", "heat_transfer", "fuel_combustion_heat"])
        if any(w in p_lower for w in ["ác-si-mét", "ac si met", "nhúng chìm", "lực đẩy", "chìm hoàn toàn", "buoyant", "trọng lượng riêng"]):
            matched_ids.extend(["archimedes_buoyancy", "force_gravity_mg", "weight_formula", "weight_P", "density_mass_volume", "specific_weight_from_density"])
        if any(w in p_lower for w in ["khối lượng riêng", "hình hộp", "kích thước", "thể tích", "density"]):
            matched_ids.extend(["volume_box_def", "density_mass_volume", "specific_weight_from_density", "force_gravity_mg"])

        results: list[Equation] = []
        seen = set()
        for eq_id in matched_ids:
            if eq_id not in seen:
                eq = self.kb.get_equation(eq_id)
                if eq:
                    results.append(eq)
                    seen.add(eq_id)

        # Fallback to symbol matching if needed
        if not results and given_symbols and target_symbol:
            results = self.suggest_equations(given_symbols, target_symbol)

        return results
