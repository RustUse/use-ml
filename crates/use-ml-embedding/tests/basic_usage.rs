use use_ml_embedding::{
    EmbeddingDimension, EmbeddingError, EmbeddingModelName, EmbeddingVectorShape,
};

#[test]
fn validates_embedding_metadata() -> Result<(), EmbeddingError> {
    let model = EmbeddingModelName::new("text-embedding")?;
    let dimension = EmbeddingDimension::new(384)?;
    let shape = EmbeddingVectorShape::new(dimension);

    assert_eq!(model.as_str(), "text-embedding");
    assert_eq!(shape.dimension().get(), 384);
    Ok(())
}
