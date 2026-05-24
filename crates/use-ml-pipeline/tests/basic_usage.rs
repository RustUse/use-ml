use use_ml_pipeline::{MlPipelineError, MlPipelineName, MlPipelineStatus, MlPipelineStepKind};

#[test]
fn validates_pipeline_metadata() -> Result<(), MlPipelineError> {
    let name = MlPipelineName::new("training-pipeline")?;
    let step: MlPipelineStepKind = "featurize".parse()?;

    assert_eq!(name.as_str(), "training-pipeline");
    assert_eq!(step, MlPipelineStepKind::Featurize);
    assert_eq!(MlPipelineStatus::Ready.as_str(), "ready");
    Ok(())
}
