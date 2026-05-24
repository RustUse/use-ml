use use_ml_pipeline::{MlPipelineName, MlPipelineStatus, MlPipelineStepKind};

fn main() -> Result<(), use_ml_pipeline::MlPipelineError> {
    let name = MlPipelineName::new("training-pipeline")?;
    let step: MlPipelineStepKind = "featurize".parse()?;

    assert_eq!(name.as_str(), "training-pipeline");
    assert_eq!(step, MlPipelineStepKind::Featurize);
    assert_eq!(MlPipelineStatus::Ready.as_str(), "ready");
    Ok(())
}
