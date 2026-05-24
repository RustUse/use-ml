# RustUse/use-ml

Composable machine-learning workflow primitives for `RustUse`.

`use-ml` is a focused RustUse set for representing machine-learning datasets,
features, labels, tensor shape metadata, model artifacts, training metadata,
inference metadata, evaluation metadata, metrics, pipelines, embeddings,
experiments, and model documentation as small Rust primitives. It is not a
machine-learning framework, tensor engine, autograd system, training runtime,
inference server, model registry, experiment tracker, data labeling platform, or
MLOps product.

## Workspace crates

| Crate               | Path                        | Purpose                                                                               |
| ------------------- | --------------------------- | ------------------------------------------------------------------------------------- |
| `use-ml`            | `crates/use-ml/`            | Facade over the focused machine-learning primitive crates                             |
| `use-ml-dataset`    | `crates/use-ml-dataset/`    | Dataset identity, split, provenance, and reference metadata                           |
| `use-ml-feature`    | `crates/use-ml-feature/`    | Feature names, roles, sources, encodings, transforms, and drift labels                |
| `use-ml-label`      | `crates/use-ml-label/`      | Labels, targets, classes, annotations, and quality metadata                           |
| `use-ml-tensor`     | `crates/use-ml-tensor/`     | Tensor shape, rank, dtype, layout, device, and memory-format metadata                 |
| `use-ml-model`      | `crates/use-ml-model/`      | Model identity, task, architecture, artifact, format, and lifecycle labels            |
| `use-ml-training`   | `crates/use-ml-training/`   | Training run, optimizer, loss, checkpoint, and hyperparameter metadata                |
| `use-ml-inference`  | `crates/use-ml-inference/`  | Generic inference request, prediction, serving, batching, latency, and score metadata |
| `use-ml-evaluation` | `crates/use-ml-evaluation/` | Evaluation run, validation strategy, threshold, slice, and benchmark metadata         |
| `use-ml-metric`     | `crates/use-ml-metric/`     | Metric names, values, directionality, aggregation, and common metric labels           |
| `use-ml-pipeline`   | `crates/use-ml-pipeline/`   | Pipeline, step, dependency, artifact, trigger, schedule, and run metadata             |
| `use-ml-embedding`  | `crates/use-ml-embedding/`  | Embedding model, vector, dimension, distance, index, and format metadata              |
| `use-ml-experiment` | `crates/use-ml-experiment/` | Experiment, run, parameter, artifact, backend, and stage metadata                     |
| `use-ml-model-card` | `crates/use-ml-model-card/` | Model-card section, audience, intended-use, limitation, risk, and ownership metadata  |

## Experimental

Every crate in this workspace is experimental while the release line remains
below `0.3.0`. Expect small API adjustments as the machine-learning primitive
surface settles.

## Relationship to use-ai

`use-ml` models machine-learning primitives: datasets, features, labels,
tensors, model artifacts, training, inference, evaluation, metrics, pipelines,
embeddings, experiments, and model documentation.

`use-ai` models AI interaction primitives: prompts, messages, roles, context
windows, tool calls, agents, RAG, reasoning, memory, guardrails, AI model
interfaces, and AI-specific evaluation.

These sets are siblings. They should interoperate conceptually but avoid
dependency cycles.

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

- Validated identifiers, names, labels, finite values, and bounded scores.
- Small enums for ML workflow state, artifact kinds, metric labels, model tasks,
  tensor metadata, and lifecycle metadata.
- Lightweight helpers for tensor shape metadata, metric direction labels, and
  simple value validation.
- Dependency-light primitives suitable for application glue code, docs tooling,
  examples, CLIs, and test fixtures.

## v0.2 follow-ups

Keep future `use-ml` work focused on machine-learning artifact and workflow
metadata. Good follow-up candidates include optional `serde` support, richer
metric calculation helpers, richer tensor-shape compatibility helpers,
ONNX/model-format metadata primitives, data drift and model monitoring
primitives, feature store primitives, calibration primitives, confusion matrix
helpers, fairness and bias metric labels, vector database metadata labels, model
registry metadata primitives, and optional `no_std` compatibility where
practical.

Prompt and LLM evaluation primitives, agent evaluation, RAG evaluation,
guardrail evaluation, chat model interface metadata, context-window metadata,
and tool-call metadata belong in the sibling `use-ai` set.

## Non-goals

- Training models, serving models, running inference, implementing autograd, or
  allocating tensor data.
- Feature engineering, data loading, labeling platforms, vector search,
  orchestration engines, model registries, or experiment tracking products.
- Prompt APIs, chat completions, message roles, tool calls, agents, RAG chains,
  context windows, guardrails, or AI-specific evaluation.
- Shelling out to Python, PyTorch, TensorFlow, scikit-learn, MLflow, Jupyter,
  CUDA, ONNX Runtime, package registries, model hubs, or cloud services.

## Development

```sh
cargo fmt
cargo clippy --workspace --all-targets --all-features
cargo test --workspace --all-features
```

For release-readiness checks, also run the repository tasks in the `Makefile`.

## License

Licensed under either of the following, at your option:

- Apache License, Version 2.0
- MIT license
