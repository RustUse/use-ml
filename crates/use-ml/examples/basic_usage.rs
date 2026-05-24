use use_ml::{MlDatasetName, MlFeatureName, MlModelName, TensorShape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = MlDatasetName::new("iris")?;
    let feature = MlFeatureName::new("sepal_width")?;
    let model = MlModelName::new("baseline-classifier")?;
    let shape = TensorShape::new([150, 4])?;

    assert_eq!(dataset.as_str(), "iris");
    assert_eq!(feature.as_str(), "sepal_width");
    assert_eq!(model.as_str(), "baseline-classifier");
    assert_eq!(shape.dims(), &[150, 4]);
    Ok(())
}
