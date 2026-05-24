use use_ml::{
    EmbeddingDimension, EmbeddingModelName, MlBatchSize, MlConfidenceScore, MlDatasetName,
    MlEvaluationRunId, MlExperimentName, MlFeatureName, MlLabelName, MlMetricName, MlMetricValue,
    MlModelCardName, MlModelName, MlPipelineName, TensorShape,
};

#[test]
fn facade_reexports_every_child_crate() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MlDatasetName::new("iris")?;
    let feature = MlFeatureName::new("sepal_width")?;
    let label = MlLabelName::new("species")?;
    let shape = TensorShape::new([150, 4])?;
    let model = MlModelName::new("baseline-classifier")?;
    let batch_size = MlBatchSize::new(32)?;
    let confidence = MlConfidenceScore::new(0.92)?;
    let evaluation = MlEvaluationRunId::new("eval-001")?;
    let metric = MlMetricName::new("accuracy")?;
    let metric_value = MlMetricValue::new(0.91)?;
    let pipeline = MlPipelineName::new("training-pipeline")?;
    let embedding_model = EmbeddingModelName::new("text-embedding")?;
    let dimension = EmbeddingDimension::new(384)?;
    let experiment = MlExperimentName::new("baseline")?;
    let card = MlModelCardName::new("baseline-card")?;

    assert_eq!(dataset.as_str(), "iris");
    assert_eq!(feature.as_str(), "sepal_width");
    assert_eq!(label.as_str(), "species");
    assert_eq!(shape.rank(), 2);
    assert_eq!(model.as_str(), "baseline-classifier");
    assert_eq!(batch_size.get(), 32);
    assert_eq!(confidence.value(), 0.92);
    assert_eq!(evaluation.as_str(), "eval-001");
    assert_eq!(metric.as_str(), "accuracy");
    assert_eq!(metric_value.value(), 0.91);
    assert_eq!(pipeline.as_str(), "training-pipeline");
    assert_eq!(embedding_model.as_str(), "text-embedding");
    assert_eq!(dimension.get(), 384);
    assert_eq!(experiment.as_str(), "baseline");
    assert_eq!(card.as_str(), "baseline-card");
    Ok(())
}
