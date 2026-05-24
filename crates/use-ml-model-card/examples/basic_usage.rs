use use_ml_model_card::{MlModelCard, MlModelCardName, MlModelCardSection};

fn main() -> Result<(), use_ml_model_card::MlModelCardError> {
    let card = MlModelCard::new(MlModelCardName::new("baseline-card")?)
        .with_section(MlModelCardSection::Overview);

    assert_eq!(card.name().as_str(), "baseline-card");
    assert_eq!(card.sections(), &[MlModelCardSection::Overview]);
    Ok(())
}
