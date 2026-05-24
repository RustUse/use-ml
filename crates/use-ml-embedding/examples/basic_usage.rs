use use_ml_embedding::{EmbeddingDimension, EmbeddingModelName, EmbeddingVectorShape};

fn main() -> Result<(), use_ml_embedding::EmbeddingError> {
    let model = EmbeddingModelName::new("text-embedding")?;
    let dimension = EmbeddingDimension::new(384)?;
    let shape = EmbeddingVectorShape::new(dimension);

    assert_eq!(model.as_str(), "text-embedding");
    assert_eq!(shape.dimension().get(), 384);
    Ok(())
}
