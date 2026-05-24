# use-ml-training

Training run and hyperparameter metadata primitives for `RustUse`.

## Experimental

`use-ml-training` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_training::{MlBatchSize, MlLearningRate, MlOptimizerKind, MlTrainingRunId};

let run_id = MlTrainingRunId::new("run-001")?;
let batch_size = MlBatchSize::new(32)?;
let learning_rate = MlLearningRate::new(0.001)?;
let optimizer: MlOptimizerKind = "adamw".parse()?;

assert_eq!(run_id.as_str(), "run-001");
assert_eq!(batch_size.get(), 32);
assert_eq!(learning_rate.value(), 0.001);
assert_eq!(optimizer, MlOptimizerKind::AdamW);
# Ok::<(), use_ml_training::MlTrainingError>(())
```

## Scope

- Training run IDs, job names, status, phases, optimizer labels, and loss labels.
- Positive batch sizes and epoch counts.
- Finite positive learning rates and hyperparameter names/values.

## Non-goals

- Performing training, tuning, checkpointing, or model export.
- Agent training loops, prompt optimization, RLHF, RLAIF, or instruction-tuning-specific APIs in v0.1.

## License

Licensed under either Apache-2.0 or MIT.
