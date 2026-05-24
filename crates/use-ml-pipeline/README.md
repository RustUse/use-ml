# use-ml-pipeline

ML pipeline metadata primitives for `RustUse`.

## Experimental

`use-ml-pipeline` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_pipeline::{MlPipelineName, MlPipelineStepKind, MlPipelineStatus};

let name = MlPipelineName::new("training-pipeline")?;
let step: MlPipelineStepKind = "featurize".parse()?;

assert_eq!(name.as_str(), "training-pipeline");
assert_eq!(step, MlPipelineStepKind::Featurize);
assert_eq!(MlPipelineStatus::Ready.as_str(), "ready");
# Ok::<(), use_ml_pipeline::MlPipelineError>(())
```

## Scope

- Pipeline names, identifiers, step names, step kinds, statuses, run IDs, and lifecycle labels.
- Artifact, dependency, trigger, and schedule kind metadata.

## Non-goals

- Pipeline orchestration, scheduling, deployment, or monitoring engines.
- Agent workflows, tool-routing, prompt chains, RAG chains, or reasoning plans.

## License

Licensed under either Apache-2.0 or MIT.
