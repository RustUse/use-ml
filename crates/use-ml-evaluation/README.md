# use-ml-evaluation

Evaluation run and validation metadata primitives for `RustUse`.

## Experimental

`use-ml-evaluation` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_evaluation::{MlEvaluationKind, MlEvaluationRunId, MlThreshold};

let run_id = MlEvaluationRunId::new("eval-001")?;
let threshold = MlThreshold::new(0.5)?;
let kind: MlEvaluationKind = "cross-validation".parse()?;

assert_eq!(run_id.as_str(), "eval-001");
assert_eq!(threshold.value(), 0.5);
assert_eq!(kind, MlEvaluationKind::CrossValidation);
# Ok::<(), use_ml_evaluation::MlEvaluationError>(())
```

## Scope

- Evaluation run IDs, kinds, validation strategies, targets, statuses, slices, and benchmarks.
- Threshold metadata and confusion-matrix shape metadata.
- Generic ML evaluation metadata only.

## Non-goals

- Computing evaluation metrics beyond trivial metadata validation.
- LLM-as-judge, prompt evaluation, conversation evaluation, hallucination checks, safety/guardrail evaluation, or retrieval-groundedness evaluation.

## License

Licensed under either Apache-2.0 or MIT.
