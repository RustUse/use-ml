use use_ml_feature::{MlFeatureKind, MlFeatureName, MlFeatureRole};

fn main() -> Result<(), use_ml_feature::MlFeatureError> {
    let name = MlFeatureName::new("sepal_width")?;
    let kind: MlFeatureKind = "numeric".parse()?;

    assert_eq!(name.as_str(), "sepal_width");
    assert_eq!(kind, MlFeatureKind::Numeric);
    assert_eq!(MlFeatureRole::Input.as_str(), "input");
    Ok(())
}
