use use_ml_evaluation::{MlEvaluationKind, MlEvaluationRunId, MlThreshold};

fn main() -> Result<(), use_ml_evaluation::MlEvaluationError> {
    let run_id = MlEvaluationRunId::new("eval-001")?;
    let threshold = MlThreshold::new(0.5)?;
    let kind: MlEvaluationKind = "cross-validation".parse()?;

    assert_eq!(run_id.as_str(), "eval-001");
    assert_eq!(threshold.value(), 0.5);
    assert_eq!(kind, MlEvaluationKind::CrossValidation);
    Ok(())
}
