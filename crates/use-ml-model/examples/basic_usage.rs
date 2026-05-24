use use_ml_model::{MlModelName, MlModelStage, MlModelTask};

fn main() -> Result<(), use_ml_model::MlModelError> {
    let name = MlModelName::new("baseline-classifier")?;
    let task: MlModelTask = "classification".parse()?;

    assert_eq!(name.as_str(), "baseline-classifier");
    assert_eq!(task, MlModelTask::Classification);
    assert_eq!(MlModelStage::Experimental.as_str(), "experimental");
    Ok(())
}
