use use_ml_model::{MlModelError, MlModelName, MlModelStage, MlModelTask};

#[test]
fn validates_model_metadata() -> Result<(), MlModelError> {
    let name = MlModelName::new("baseline-classifier")?;
    let task: MlModelTask = "classification".parse()?;

    assert_eq!(name.as_str(), "baseline-classifier");
    assert_eq!(task, MlModelTask::Classification);
    assert_eq!(MlModelStage::Experimental.as_str(), "experimental");
    Ok(())
}
