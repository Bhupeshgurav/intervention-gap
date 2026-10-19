"""Numerical audit independent of any trained model or model ranking."""
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

from research.process.cstr import cstr_rhs, simulate_commands, REFERENCE_EQUILIBRIUM


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "results" / "process_validation"
    out.mkdir(parents=True, exist_ok=True)
    source = ast.parse((root / "vendor/pcgym_reference/model_classes.py").read_text())
    selected = ast.Module(body=[node for node in source.body if isinstance(node, ast.ClassDef) and node.name in ("BaseModel", "cstr")], type_ignores=[])
    namespace = {"dataclass": dataclass, "np": np, "jnp": np}
    exec(compile(selected, "pinned_upstream_cstr_classes", "exec"), namespace)
    upstream = namespace["cstr"](int_method="numpy")
    states = np.array([[c, t] for c in [.72, .84, .96] for t in [315, 320, 325, 330, 335, 340, 345] for _ in range(3)])
    commands = np.tile([290., 296., 302.], len(states) // 3)
    derivative_error = float(np.max(np.abs(cstr_rhs(states, commands) - np.array([upstream(x, np.array([u])) for x, u in zip(states, commands)]))))
    assert derivative_error < 1e-10
    times = np.linspace(0, 1, 21)
    reference, reference_records = [], []
    for x, u in zip(states, commands):
        sol = solve_ivp(lambda t, y: upstream(y, np.array([u])), (0, 1), x, method="Radau", t_eval=times, rtol=1e-10, atol=1e-12)
        assert sol.success and sol.y.shape == (2, 21)
        reference.append(sol.y.T)
        reference_records.append({"initial_state": x.tolist(), "command": float(u), "max_temperature_K": float(sol.y[1].max()), "min_concentration": float(sol.y[0].min())})
    reference = np.array(reference)
    comparisons = []
    for step in [.005, .001, .0005, .00025, .000125, .0000625]:
        start = time.perf_counter()
        try:
            trajectory = simulate_commands(states, np.repeat(commands[:, None], 20, axis=1), dt=.05, max_step=step)
            error = np.abs(trajectory - reference)
            comparisons.append({"max_step": step, "success": True, "max_abs_error_concentration": float(error[..., 0].max()), "max_abs_error_temperature_K": float(error[..., 1].max()), "seconds": time.perf_counter() - start})
        except FloatingPointError as exc:
            comparisons.append({"max_step": step, "success": False, "error": str(exc), "seconds": time.perf_counter() - start})
    report = {"scope": "Development-only physical/numerical audit; not a learned-model experiment", "derivative_max_error": derivative_error, "equilibrium_derivative": cstr_rhs(REFERENCE_EQUILIBRIUM, 300).tolist(), "reference_solver": "SciPy Radau rtol=1e-10 atol=1e-12 applied to pinned upstream derivative", "cases": reference_records, "rk4_comparison": comparisons}
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(out / "reference_trajectories.npz", initial_states=states, commands=commands, times=times, trajectories=reference)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
