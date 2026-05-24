#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub const MAX_TENSOR_RANK: usize = 64;

pub mod prelude {
    pub use crate::{
        MAX_TENSOR_RANK, TensorAxis, TensorDType, TensorDeviceKind, TensorDim, TensorLayout,
        TensorMemoryFormat, TensorRank, TensorShape, TensorShapeError,
    };
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorShape {
    dims: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self, TensorShapeError> {
        let dims = dims.into();
        if dims.len() > MAX_TENSOR_RANK {
            return Err(TensorShapeError::RankTooLarge {
                rank: dims.len(),
                max: MAX_TENSOR_RANK,
            });
        }

        Ok(Self { dims })
    }

    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn num_elements(&self) -> Option<usize> {
        self.dims
            .iter()
            .copied()
            .try_fold(1_usize, usize::checked_mul)
    }

    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    pub fn is_vector(&self) -> bool {
        self.rank() == 1
    }

    pub fn is_matrix(&self) -> bool {
        self.rank() == 2
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorDim(usize);

impl TensorDim {
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

impl fmt::Display for TensorDim {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorRank(usize);

impl TensorRank {
    pub fn new(value: usize) -> Result<Self, TensorShapeError> {
        if value > MAX_TENSOR_RANK {
            Err(TensorShapeError::RankTooLarge {
                rank: value,
                max: MAX_TENSOR_RANK,
            })
        } else {
            Ok(Self(value))
        }
    }

    pub const fn get(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TensorAxis(usize);

impl TensorAxis {
    pub fn new(value: usize, rank: TensorRank) -> Result<Self, TensorShapeError> {
        if value < rank.get() {
            Ok(Self(value))
        } else {
            Err(TensorShapeError::AxisOutOfBounds {
                axis: value,
                rank: rank.get(),
            })
        }
    }

    pub const fn unchecked(value: usize) -> Self {
        Self(value)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

macro_rules! tensor_enum {
    ($name:ident { $($variant:ident => $label:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $label),+
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = TensorShapeError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(TensorShapeError::UnknownLabel),
                }
            }
        }
    };
}

tensor_enum!(TensorDType {
    Bool => "bool",
    Int8 => "int8",
    Int16 => "int16",
    Int32 => "int32",
    Int64 => "int64",
    Uint8 => "uint8",
    Uint16 => "uint16",
    Uint32 => "uint32",
    Uint64 => "uint64",
    Float16 => "float16",
    BFloat16 => "bfloat16",
    Float32 => "float32",
    Float64 => "float64",
    Complex64 => "complex64",
    Complex128 => "complex128",
    String => "string",
    Unknown => "unknown",
});

tensor_enum!(TensorLayout {
    Dense => "dense",
    Sparse => "sparse",
    Ragged => "ragged",
    Quantized => "quantized",
    BlockSparse => "block-sparse",
    Unknown => "unknown",
});

tensor_enum!(TensorDeviceKind {
    Cpu => "cpu",
    Gpu => "gpu",
    Tpu => "tpu",
    Npu => "npu",
    Metal => "metal",
    Vulkan => "vulkan",
    Wasm => "wasm",
    Unknown => "unknown",
});

tensor_enum!(TensorMemoryFormat {
    Contiguous => "contiguous",
    ChannelsLast => "channels-last",
    ChannelsFirst => "channels-first",
    Sparse => "sparse",
    Unknown => "unknown",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorShapeError {
    EmptyLabel,
    UnknownLabel,
    RankTooLarge { rank: usize, max: usize },
    AxisOutOfBounds { axis: usize, rank: usize },
}

impl fmt::Display for TensorShapeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLabel => formatter.write_str("tensor metadata label cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown tensor metadata label"),
            Self::RankTooLarge { rank, max } => {
                write!(formatter, "tensor rank {rank} exceeds maximum rank {max}")
            },
            Self::AxisOutOfBounds { axis, rank } => {
                write!(formatter, "tensor axis {axis} is outside rank {rank}")
            },
        }
    }
}

impl Error for TensorShapeError {}

fn normalized_label(value: &str) -> Result<String, TensorShapeError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(TensorShapeError::EmptyLabel)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        TensorAxis, TensorDType, TensorDeviceKind, TensorLayout, TensorMemoryFormat, TensorRank,
        TensorShape, TensorShapeError,
    };

    #[test]
    fn models_tensor_shapes() -> Result<(), TensorShapeError> {
        let scalar = TensorShape::scalar();
        let vector = TensorShape::new([3])?;
        let matrix = TensorShape::new([2, 3])?;
        let tensor = TensorShape::new([2, 3, 4])?;

        assert!(scalar.is_scalar());
        assert!(vector.is_vector());
        assert!(matrix.is_matrix());
        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.dims(), &[2, 3, 4]);
        assert_eq!(tensor.num_elements(), Some(24));
        Ok(())
    }

    #[test]
    fn protects_element_count_overflow() -> Result<(), TensorShapeError> {
        let shape = TensorShape::new([usize::MAX, 2])?;

        assert_eq!(shape.num_elements(), None);
        Ok(())
    }

    #[test]
    fn validates_rank_and_axis() -> Result<(), TensorShapeError> {
        let rank = TensorRank::new(3)?;
        let axis = TensorAxis::new(2, rank)?;

        assert_eq!(rank.get(), 3);
        assert_eq!(axis.index(), 2);
        assert_eq!(
            TensorAxis::new(3, rank),
            Err(TensorShapeError::AxisOutOfBounds { axis: 3, rank: 3 })
        );
        Ok(())
    }

    #[test]
    fn displays_and_parses_tensor_enums() -> Result<(), TensorShapeError> {
        assert_eq!("float32".parse::<TensorDType>()?, TensorDType::Float32);
        assert_eq!(
            "block sparse".parse::<TensorLayout>()?,
            TensorLayout::BlockSparse
        );
        assert_eq!("gpu".parse::<TensorDeviceKind>()?, TensorDeviceKind::Gpu);
        assert_eq!(
            "channels_last".parse::<TensorMemoryFormat>()?,
            TensorMemoryFormat::ChannelsLast
        );
        Ok(())
    }
}
