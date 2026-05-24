# use-ml

Facade crate for the focused machine-learning primitive crates in `RustUse`.

## Experimental

`use-ml` is experimental while the workspace remains below `0.3.0`.

## Example

```rust
use use_ml::{MlDatasetName, MlFeatureName, MlModelName, TensorShape};

let dataset = MlDatasetName::new("iris")?;
let feature = MlFeatureName::new("sepal_width")?;
let model = MlModelName::new("baseline-classifier")?;
let shape = TensorShape::new([150, 4])?;

assert_eq!(dataset.as_str(), "iris");
assert_eq!(feature.as_str(), "sepal_width");
assert_eq!(model.as_str(), "baseline-classifier");
assert_eq!(shape.rank(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Scope

- Re-export the focused `use-ml-*` primitive crates.
- Keep implementation logic inside focused child crates.
- Provide one dependency for machine-learning metadata primitives.

## Relationship to use-ai

`use-ml` models machine-learning primitives: datasets, features, labels,
tensors, model artifacts, training, inference, evaluation, metrics, pipelines,
embeddings, experiments, and model documentation.

`use-ai` models AI interaction primitives: prompts, messages, roles, context
windows, tool calls, agents, RAG, reasoning, memory, guardrails, AI model
interfaces, and AI-specific evaluation.

These sets are siblings. They should interoperate conceptually but avoid
dependency cycles.

## Non-goals

- Training, inference, serving, tensor math, vector search, registry behavior, or experiment tracking.
- Prompt, chat, agent, RAG, guardrail, or AI-provider interface modeling.

## License

Licensed under either Apache-2.0 or MIT.
