# use-ml-metric

Metric name, value, and direction metadata primitives for `RustUse`.

## Experimental

`use-ml-metric` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_metric::{MlClassificationMetric, MlMetricDirection, MlMetricName, MlMetricValue};

let name = MlMetricName::new("accuracy")?;
let value = MlMetricValue::new(0.93)?;

assert_eq!(name.as_str(), "accuracy");
assert_eq!(value.value(), 0.93);
assert_eq!(MlClassificationMetric::Accuracy.direction(), MlMetricDirection::HigherIsBetter);
# Ok::<(), use_ml_metric::MlMetricError>(())
```

## Scope

- Metric names, finite metric values, directions, aggregations, and metric categories.
- Generic classification, regression, ranking, clustering, and generation metric labels.
- Obvious metric-direction helpers.

## Non-goals

- Complete metric calculation implementations in v0.1.
- LLM-specific metrics such as faithfulness, groundedness, toxicity, refusal quality, jailbreak resistance, instruction following, or answer relevance.

## License

Licensed under either Apache-2.0 or MIT.
