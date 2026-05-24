# use-ml-experiment

Experiment tracking metadata primitives for `RustUse`.

## Experimental

`use-ml-experiment` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_experiment::{MlExperimentName, MlRunId, MlRunStatus};

let experiment = MlExperimentName::new("baseline")?;
let run = MlRunId::new("run-001")?;

assert_eq!(experiment.as_str(), "baseline");
assert_eq!(run.as_str(), "run-001");
assert_eq!(MlRunStatus::Finished.as_str(), "finished");
# Ok::<(), use_ml_experiment::MlExperimentError>(())
```

## Scope

- Experiment names, experiment IDs, run IDs, run statuses, tags, parameters, artifact URIs, backend labels, and stages.
- Metadata only; no tracking server behavior.

## Non-goals

- Contacting tracking servers or implementing MLflow-like systems.
- Prompt experiments, agent experiments, conversation evaluations, or LLM evaluation suites.

## License

Licensed under either Apache-2.0 or MIT.
