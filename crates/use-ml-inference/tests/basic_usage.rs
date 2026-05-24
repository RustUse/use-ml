use use_ml_inference::{
    MlConfidenceScore, MlInferenceError, MlInferenceMode, MlInferenceRequestId,
};

#[test]
fn validates_inference_metadata() -> Result<(), MlInferenceError> {
    let request = MlInferenceRequestId::new("req-001")?;
    let score = MlConfidenceScore::new(0.87)?;
    let mode: MlInferenceMode = "online".parse()?;

    assert_eq!(request.as_str(), "req-001");
    assert_eq!(score.value(), 0.87);
    assert_eq!(mode, MlInferenceMode::Online);
    Ok(())
}
