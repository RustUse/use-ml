use use_ml_tensor::{TensorDType, TensorShape};

fn main() -> Result<(), use_ml_tensor::TensorShapeError> {
    let shape = TensorShape::new([2, 3, 4])?;

    assert_eq!(shape.rank(), 3);
    assert_eq!(shape.num_elements(), Some(24));
    assert_eq!(TensorDType::Float32.as_str(), "float32");
    Ok(())
}
