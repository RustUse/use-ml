use use_ml_label::{MlClassName, MlLabelError, MlLabelName, MlTargetKind};

#[test]
fn validates_label_metadata() -> Result<(), MlLabelError> {
    let label = MlLabelName::new("species")?;
    let class = MlClassName::new("setosa")?;
    let target: MlTargetKind = "multiclass-classification".parse()?;

    assert_eq!(label.as_str(), "species");
    assert_eq!(class.as_str(), "setosa");
    assert_eq!(target, MlTargetKind::MulticlassClassification);
    Ok(())
}
