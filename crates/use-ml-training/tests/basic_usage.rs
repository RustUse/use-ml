use use_ml_training::{
    MlBatchSize, MlLearningRate, MlOptimizerKind, MlTrainingError, MlTrainingRunId,
};

#[test]
fn validates_training_metadata() -> Result<(), MlTrainingError> {
    let run_id = MlTrainingRunId::new("run-001")?;
    let batch_size = MlBatchSize::new(32)?;
    let learning_rate = MlLearningRate::new(0.001)?;
    let optimizer: MlOptimizerKind = "adamw".parse()?;

    assert_eq!(run_id.as_str(), "run-001");
    assert_eq!(batch_size.get(), 32);
    assert_eq!(learning_rate.value(), 0.001);
    assert_eq!(optimizer, MlOptimizerKind::AdamW);
    Ok(())
}
