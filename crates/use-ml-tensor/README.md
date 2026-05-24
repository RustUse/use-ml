# use-ml-tensor

Tensor shape and metadata primitives for `RustUse` machine-learning workflows.

## Experimental

`use-ml-tensor` is experimental while `use-ml` remains below `0.3.0`.

## Example

```rust
use use_ml_tensor::{TensorDType, TensorShape};

let shape = TensorShape::new([2, 3, 4])?;

assert_eq!(shape.rank(), 3);
assert_eq!(shape.num_elements(), Some(24));
assert_eq!(TensorDType::Float32.as_str(), "float32");
# Ok::<(), use_ml_tensor::TensorShapeError>(())
```

## Scope

- Tensor shape, dimension, rank, and axis metadata.
- Dtype, layout, device-kind, and memory-format labels.
- Checked shape helpers such as rank and element-count metadata.

## Non-goals

- Tensor math, autograd, model execution, tensor allocation, or framework bindings.
- Device APIs, runtime scheduling, kernels, CUDA, ONNX Runtime, or data movement.

## License

Licensed under either Apache-2.0 or MIT.
