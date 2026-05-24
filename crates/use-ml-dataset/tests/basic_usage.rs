use use_ml_dataset::{MlDatasetError, MlDatasetName, MlDatasetSplit, MlDatasetVersion};

#[test]
fn validates_dataset_metadata() -> Result<(), MlDatasetError> {
    let name = MlDatasetName::new("iris")?;
    let version = MlDatasetVersion::new("v1")?;
    let split: MlDatasetSplit = "train".parse()?;

    assert_eq!(name.as_str(), "iris");
    assert_eq!(version.as_str(), "v1");
    assert_eq!(split.to_string(), "train");
    Ok(())
}
