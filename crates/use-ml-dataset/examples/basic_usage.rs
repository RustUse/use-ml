use use_ml_dataset::{MlDatasetName, MlDatasetProvenance, MlDatasetSplit};

fn main() -> Result<(), use_ml_dataset::MlDatasetError> {
    let name = MlDatasetName::new("iris")?;
    let split: MlDatasetSplit = "validation".parse()?;

    assert_eq!(name.as_str(), "iris");
    assert_eq!(split.to_string(), "validation");
    assert_eq!(MlDatasetProvenance::HumanCreated.as_str(), "human-created");
    Ok(())
}
