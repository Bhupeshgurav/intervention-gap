# Action/outcome record checker

A small executable demonstration of data qualification, not a causal effect estimator. It accepts newline-delimited JSON and writes a report retaining every input record and rejection reason. `examples/synthetic_action_records.jsonl` is entirely fabricated and contains no company or factory data.

```sh
python -m research.records.qualify examples/synthetic_action_records.jsonl --output results/record_quality_demo/report.json
```

The example has seven records. Two pass the declared consistency checks, one of which records a requested/applied action mismatch. Other records demonstrate future-available features, unknown execution, unresolved censoring, and duplicate episode identifiers. None of those figures is a customer result.

## Fields

- `schema_version`: 1.
- `episode_id`, `plant_id`, `machine_id`, `context_label`: nonempty identifiers. The context label must be defined independently of future outcomes; the checker cannot verify its semantic provenance.
- `decision_at`: timezone-aware ISO timestamp.
- `observations`: a nonempty mapping of feature names to numeric `value` and `available_at` timestamp. Availability must not follow the decision.
- `requested_action`: `action_id` and `requested_at`.
- `execution`: `status`, plus `action_id` and `applied_at` for executed records. Unknown or rejected execution does not qualify for this executed-action evaluation view, but remains in the report.
- `outcome`: numeric `value`, `metric`, `unit`, `observed_at`, and explicit `censored: false` for this complete-outcome view.
- `provenance`: source and label method.

Requested and applied actions are never silently equated. A mismatch creates a warning, and coverage is counted by the applied action. Coverage groups also preserve machine, context, outcome metric, and unit; unrelated outcomes are not pooled.

## What eligibility means

Eligibility means only that the record passes these declared structural and time-order checks. It does not establish correct labels, causal attribution, adequate observation history, action comparability, independent episodes, appropriate outcome windows, sufficient samples, or a safe intervention. A recommendation must not be released merely because its records pass.

Dropping unknown/rejected executions or censored outcomes from a causal estimate can introduce selection bias. Their counts must be discussed, not hidden by the eligible view. Treatment-policy effects may require including nonexecution and intention-to-treat analysis; this checker does not implement that analysis.

This is an initial schema. Real factory integration requires agreement on synchronized clocks, execution acknowledgement semantics, event grouping, relevant context, recurrence windows, concurrent actions, censoring, and data rights. The example action identifiers are fictional labels, not operational instructions.
