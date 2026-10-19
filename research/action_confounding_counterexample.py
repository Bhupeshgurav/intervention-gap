"""Exact illustrative counterexample; not an empirical industrial result.

S is unobserved fault severity, A is a binary corrective action, and Y is
an illustrative downtime outcome. P(S=1)=1/2; P(A=S)=q;
Y=5+10*S-2*A. Observed telemetry is uninformative in this construction.

This enumerates Bayes-optimal observational predictors, not trained models.
It illustrates established confounding, not a new causality theorem.
Run: python3 research/action_confounding_counterexample.py
"""

from fractions import Fraction as F
import json
from pathlib import Path


def evaluate(q):
    rows = [
        (s, a, F(1, 2) * (q if a == s else 1 - q), 5 + 10 * s - 2 * a)
        for s in (0, 1)
        for a in (0, 1)
    ]
    assert sum(p for _, _, p, _ in rows) == 1
    mean_y = sum(p * y for _, _, p, y in rows)
    action_means = {
        a: sum(p * y for _, action, p, y in rows if action == a)
        / sum(p for _, action, p, _ in rows if action == a)
        for a in (0, 1)
    }
    mse_observation = sum(p * (y - mean_y) ** 2 for _, _, p, y in rows)
    mse_action = sum(p * (y - action_means[a]) ** 2 for _, a, p, y in rows)
    do_means = {a: sum(F(1, 2) * (5 + 10 * s - 2 * a) for s in (0, 1)) for a in (0, 1)}
    recommended = None if action_means[0] == action_means[1] else min(action_means, key=action_means.get)
    optimum = min(do_means, key=do_means.get)
    return {
        "operator_matches_hidden_severity_probability": float(q),
        "observation_only_mse": float(mse_observation),
        "action_conditioned_mse": float(mse_action),
        "relative_mse_reduction": float(1 - mse_action / mse_observation),
        "observational_expected_y_no_action": float(action_means[0]),
        "observational_expected_y_action": float(action_means[1]),
        "observational_effect": float(action_means[1] - action_means[0]),
        "interventional_expected_y_no_action": float(do_means[0]),
        "interventional_expected_y_action": float(do_means[1]),
        "true_effect": float(do_means[1] - do_means[0]),
        "observational_recommendation": recommended,
        "optimal_intervention": optimum,
        "decision_regret": None if recommended is None else float(do_means[recommended] - do_means[optimum]),
    }


def main():
    illustrative = evaluate(F(9, 10))
    assert illustrative["observation_only_mse"] == 18.0
    assert illustrative["action_conditioned_mse"] == 9.0
    assert illustrative["observational_expected_y_no_action"] == 6.0
    assert illustrative["observational_expected_y_action"] == 12.0
    assert illustrative["true_effect"] == -2.0
    assert illustrative["observational_recommendation"] == 0
    assert illustrative["optimal_intervention"] == 1
    assert illustrative["decision_regret"] == 2.0
    randomized = evaluate(F(1, 2))
    assert randomized["observational_effect"] == randomized["true_effect"]
    assert randomized["decision_regret"] == 0.0

    report = {
        "status": "exact analytic toy enumeration; no training or real data",
        "equations": ["P(S=1)=0.5", "P(A=S)=q", "Y=5+10*S-2*A"],
        "limitations": [
            "Observed telemetry is uninformative by construction.",
            "Both actions have positive probability at each severity for q<1.",
            "This is established confounding, not a novel theorem.",
            "This does not show failure of any Celesnity model.",
        ],
        "illustrative_q_0_9": illustrative,
        "policy_sweep": [evaluate(q) for q in (F(1, 2), F(3, 5), F(7, 10), F(4, 5), F(9, 10), F(99, 100))],
    }
    output = Path(__file__).with_name("counterexample_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": "passed", "output": str(output), "illustrative_result": illustrative}, indent=2))


if __name__ == "__main__":
    main()
