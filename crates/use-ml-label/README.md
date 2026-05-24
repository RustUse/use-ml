# use-ml-label

Label and target metadata primitives for `RustUse` machine-learning workflows.

## Experimental

`use-ml-label` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_label::{MlClassName, MlLabelName, MlTargetKind};

let label = MlLabelName::new("species")?;
let class = MlClassName::new("setosa")?;
let target: MlTargetKind = "multiclass-classification".parse()?;

assert_eq!(label.as_str(), "species");
assert_eq!(class.as_str(), "setosa");
assert_eq!(target, MlTargetKind::MulticlassClassification);
# Ok::<(), use_ml_label::MlLabelError>(())
```

## Scope

- Label names, class names, identifiers, and target-kind labels.
- Annotation, source, quality, and cardinality metadata.
- Primitive label metadata only.

## Non-goals

- Data labeling platforms or annotation workflow engines.
- AI-response ratings, human preference labels, RLHF/RLAIF labels, or LLM judge labels.

## License

Licensed under either Apache-2.0 or MIT.
