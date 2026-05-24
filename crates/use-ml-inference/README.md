# use-ml-inference

Generic inference and prediction metadata primitives for `RustUse`.

## Experimental

`use-ml-inference` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_inference::{MlConfidenceScore, MlInferenceMode, MlInferenceRequestId};

let request = MlInferenceRequestId::new("req-001")?;
let score = MlConfidenceScore::new(0.87)?;
let mode: MlInferenceMode = "online".parse()?;

assert_eq!(request.as_str(), "req-001");
assert_eq!(score.value(), 0.87);
assert_eq!(mode, MlInferenceMode::Online);
# Ok::<(), use_ml_inference::MlInferenceError>(())
```

## Scope

- Inference request IDs, prediction IDs, modes, statuses, and serving labels.
- Input/output kind labels, batching labels, latency buckets, and confidence scores.
- Generic prediction metadata only.

## Non-goals

- Serving models, running inference, making network calls, or streaming outputs.
- Chat completions, prompt completions, message roles, assistant responses, system prompts, tool calls, function calls, or streaming chunks.

## License

Licensed under either Apache-2.0 or MIT.
