# Guide: Adding New Physics Equations

Adding new physics equations to the engine is simple and does not require modifying engine code.

## Step-by-Step Guide

### Step 1: Ensure All Physical Quantities Exist
Open `data/knowledge/quantities.yaml` and verify that all physical quantities used in your new equation are defined.

If a quantity is missing, append it:
```yaml
  - name: "angular_velocity"
    symbol: "omega"
    dimension: "T^-1"
    si_unit: "radian / second"
    aliases: ["rotational speed", "angular speed"]
```

### Step 2: Add Equation to the Topic YAML
Open or create a YAML file under `data/knowledge/equations/` (e.g. `data/knowledge/equations/rotational.yaml`):

```yaml
equations:
  - id: "rot_kin_vel"
    name: "Angular velocity definition"
    expression: "omega = theta / t"
    variables: ["omega", "theta", "t"]
    variable_quantities:
      omega: "angular_velocity"
      theta: "angular_displacement"
      t: "time"
    domain: "mechanics"
    topic: "rotational"
    description: "Angular velocity is angular displacement divided by time"
    conditions: ["constant angular velocity"]
    dimension_lhs: "T^-1"
    dimension_rhs: "T^-1"
```

### Step 3: Validate Knowledge Base Consistency
Run the automated validator:

```bash
uv run physics-engine validate-kb
```
or
```bash
uv run python scripts/validate_knowledge_base.py
```

The validator will automatically:
1. Verify SymPy syntactic parsing for your new equation.
2. Cross-check all variable quantity references.
3. Compute LHS and RHS dimensions using Pint to verify dimensional balance.
