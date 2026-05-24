use use_ml_metric::{
    MlClassificationMetric, MlMetricDirection, MlMetricError, MlMetricName, MlMetricValue,
};

#[test]
fn validates_metric_metadata() -> Result<(), MlMetricError> {
    let name = MlMetricName::new("accuracy")?;
    let value = MlMetricValue::new(0.93)?;

    assert_eq!(name.as_str(), "accuracy");
    assert_eq!(value.value(), 0.93);
    assert_eq!(
        MlClassificationMetric::Accuracy.direction(),
        MlMetricDirection::HigherIsBetter
    );
    Ok(())
}
