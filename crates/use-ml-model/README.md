# use-ml-model

ML model artifact metadata primitives for `RustUse`.

## Experimental

`use-ml-model` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_model::{MlModelName, MlModelTask, MlModelStage};

let name = MlModelName::new("baseline-classifier")?;
let task: MlModelTask = "classification".parse()?;

assert_eq!(name.as_str(), "baseline-classifier");
assert_eq!(task, MlModelTask::Classification);
assert_eq!(MlModelStage::Experimental.as_str(), "experimental");
# Ok::<(), use_ml_model::MlModelError>(())
```

## Scope

- Model names, identifiers, versions, generic provider labels, and license labels.
- Model kind, task, architecture, artifact, format, and lifecycle-stage labels.
- Artifact metadata only.

## Non-goals

- Loading, training, running, serving, or registering models.
- Chat/completion APIs, context windows, token limits, tool-use support, streaming support, reasoning modes, pricing, or API endpoint metadata.

## License

Licensed under either Apache-2.0 or MIT.
