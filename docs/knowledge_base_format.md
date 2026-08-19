# Physics Knowledge Base Specification

The Physics Knowledge Base stores physical quantities and equations in modular YAML format under `data/knowledge/`.

## 1. Physical Quantity Schema (`quantities.yaml`)

Each quantity is defined with the following fields:

```yaml
quantities:
  - name: "acceleration"
    symbol: "a"
    dimension: "L T^-2"
    si_unit: "meter / second ** 2"
    aliases: ["deceleration", "retardation", "rate of acceleration"]
```

### Fields:
- `name` (string, required): Canonical name in lowercase snake_case.
- `symbol` (string, required): Standard physics mathematical symbol (e.g. `F`, `v_i`, `KE`).
- `dimension` (string, required): Base SI dimensions (`M`, `L`, `T`, `I`, `Theta`, `N`, `J`) with exponents.
- `si_unit` (string, required): Pint-parseable SI unit expression.
- `aliases` (list[string], optional): Synonyms and natural language variations.

## 2. Equation Schema (`equations/*.yaml`)

Equations are categorized into domain-specific YAML files (e.g. `kinematics.yaml`, `newtons_laws.yaml`, `work_energy.yaml`).

```yaml
equations:
  - id: "newton2"
    name: "Newton's Second Law"
    expression: "F = m * a"
    variables: ["F", "m", "a"]
    variable_quantities:
      F: "force"
      m: "mass"
      a: "acceleration"
    domain: "mechanics"
    topic: "newton_laws"
    description: "Net force equals mass times acceleration"
    conditions: ["constant mass", "inertial frame"]
    dimension_lhs: "M L T^-2"
    dimension_rhs: "M L T^-2"
```

### Fields:
- `id` (string, required): Unique identifier (e.g. `kin_eq1`, `work_def`).
- `name` (string, required): Human-readable equation name.
- `expression` (string, required): SymPy-parseable equation containing exactly one `=` sign.
- `variables` (list[string], required): Variable symbols present in the expression.
- `variable_quantities` (dict[str, str], required): Mapping of each variable symbol to its quantity name in `quantities.yaml`.
- `domain` (string, required): Physics domain (`mechanics`, `electricity`, etc.).
- `topic` (string, required): Subtopic category.
- `description` (string, optional): Conceptual description of physical applicability.
- `conditions` (list[string], optional): Physical assumptions (e.g. `constant acceleration`).
- `dimension_lhs` / `dimension_rhs` (string, required): Dimensional formulas for LHS and RHS.
