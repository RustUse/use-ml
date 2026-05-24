use use_ml_metric::{MlClassificationMetric, MlMetricDirection, MlMetricName, MlMetricValue};

fn main() -> Result<(), use_ml_metric::MlMetricError> {
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
