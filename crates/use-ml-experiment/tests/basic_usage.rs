use use_ml_experiment::{MlExperimentError, MlExperimentName, MlRunId, MlRunStatus};

#[test]
fn validates_experiment_metadata() -> Result<(), MlExperimentError> {
    let experiment = MlExperimentName::new("baseline")?;
    let run = MlRunId::new("run-001")?;

    assert_eq!(experiment.as_str(), "baseline");
    assert_eq!(run.as_str(), "run-001");
    assert_eq!(MlRunStatus::Finished.as_str(), "finished");
    Ok(())
}
