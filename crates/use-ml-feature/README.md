# use-ml-feature

Feature metadata primitives for `RustUse` machine-learning workflows.

## Experimental

`use-ml-feature` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_feature::{MlFeatureKind, MlFeatureName, MlFeatureRole};

let name = MlFeatureName::new("sepal_width")?;
let kind: MlFeatureKind = "numeric".parse()?;

assert_eq!(name.as_str(), "sepal_width");
assert_eq!(kind, MlFeatureKind::Numeric);
assert_eq!(MlFeatureRole::Input.as_str(), "input");
# Ok::<(), use_ml_feature::MlFeatureError>(())
```

## Scope

- Feature names, identifiers, kinds, roles, and sources.
- Transform, encoding, scaling, missing-value, and drift-status labels.
- Lightweight validation for ASCII-safe feature names.

## Non-goals

- Performing feature engineering, transformations, or pipeline orchestration.
- Prompt variables, message fields, context-window fields, or tool-call arguments.

## License

Licensed under either Apache-2.0 or MIT.
