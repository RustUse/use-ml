# use-ml-model-card

Model-card metadata primitives for `RustUse` machine-learning workflows.

## Experimental

`use-ml-model-card` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_model_card::{MlModelCard, MlModelCardName, MlModelCardSection};

let card = MlModelCard::new(MlModelCardName::new("baseline-card")?)
    .with_section(MlModelCardSection::Overview);

assert_eq!(card.name().as_str(), "baseline-card");
assert_eq!(card.sections(), &[MlModelCardSection::Overview]);
# Ok::<(), use_ml_model_card::MlModelCardError>(())
```

## Scope

- Model-card names, sections, intended use labels, audience labels, limitation text, risk labels, evaluation summaries, dataset references, and owners.
- Metadata only for documenting ML models.

## Non-goals

- Generating full reports, making policy claims, or validating policy compliance.
- Prompt cards, assistant cards, agent cards, or AI product cards in v0.1.

## License

Licensed under either Apache-2.0 or MIT.
