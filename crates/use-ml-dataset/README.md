# use-ml-dataset

Dataset identity and metadata primitives for `RustUse` machine-learning workflows.

## Experimental

`use-ml-dataset` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_dataset::{MlDatasetName, MlDatasetSplit, MlDatasetProvenance};

let name = MlDatasetName::new("iris")?;
let split: MlDatasetSplit = "validation".parse()?;

assert_eq!(name.as_str(), "iris");
assert_eq!(split.to_string(), "validation");
assert_eq!(MlDatasetProvenance::Synthetic.as_str(), "synthetic");
# Ok::<(), use_ml_dataset::MlDatasetError>(())
```

## Scope

- Dataset names, identifiers, versions, splits, and kinds.
- Example identifiers and example-kind labels.
- Dataset license, provenance, schema-reference, and dataset-card-reference metadata.

## Non-goals

- Loading datasets or parsing full dataset schemas.
- Prompt datasets, chat logs, RAG corpora, or conversation-specific concepts unless modeled generically as datasets.

## License

Licensed under either Apache-2.0 or MIT.
