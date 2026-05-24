#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::{error::Error, num::NonZeroUsize};

pub mod prelude {
    pub use crate::{
        EmbeddingDimension, EmbeddingDistanceMetric, EmbeddingError, EmbeddingIndexKind,
        EmbeddingModality, EmbeddingModelName, EmbeddingNormalizationKind, EmbeddingSearchKind,
        EmbeddingVectorFormat, EmbeddingVectorId, EmbeddingVectorShape,
    };
}

macro_rules! embedding_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, EmbeddingError> {
                non_empty_text(value).map(Self)
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                self.as_str()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.as_str())
            }
        }

        impl FromStr for $name {
            type Err = EmbeddingError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = EmbeddingError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! embedding_enum {
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
            type Err = EmbeddingError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(EmbeddingError::UnknownLabel),
                }
            }
        }
    };
}

embedding_text_newtype!(EmbeddingModelName);
embedding_text_newtype!(EmbeddingVectorId);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmbeddingDimension(NonZeroUsize);

impl EmbeddingDimension {
    pub fn new(value: usize) -> Result<Self, EmbeddingError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(EmbeddingError::Zero)
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EmbeddingVectorShape {
    dimension: EmbeddingDimension,
}

impl EmbeddingVectorShape {
    pub const fn new(dimension: EmbeddingDimension) -> Self {
        Self { dimension }
    }

    pub const fn dimension(self) -> EmbeddingDimension {
        self.dimension
    }
}

embedding_enum!(EmbeddingModality {
    Text => "text",
    Image => "image",
    Audio => "audio",
    Video => "video",
    Code => "code",
    Tabular => "tabular",
    Graph => "graph",
    Multimodal => "multimodal",
    Other => "other",
});

embedding_enum!(EmbeddingDistanceMetric {
    Cosine => "cosine",
    DotProduct => "dot-product",
    Euclidean => "euclidean",
    Manhattan => "manhattan",
    Hamming => "hamming",
    Jaccard => "jaccard",
    Custom => "custom",
});

embedding_enum!(EmbeddingNormalizationKind {
    None => "none",
    Unit => "unit",
    MeanCentered => "mean-centered",
    Standardized => "standardized",
    Custom => "custom",
});

embedding_enum!(EmbeddingIndexKind {
    Flat => "flat",
    Hnsw => "hnsw",
    Ivf => "ivf",
    Pq => "pq",
    IvfPq => "ivf-pq",
    Annoy => "annoy",
    Scann => "scann",
    Other => "other",
});

embedding_enum!(EmbeddingSearchKind {
    Exact => "exact",
    Approximate => "approximate",
    Hybrid => "hybrid",
    Filtered => "filtered",
    Reranked => "reranked",
});

embedding_enum!(EmbeddingVectorFormat {
    Dense => "dense",
    Sparse => "sparse",
    Binary => "binary",
    Quantized => "quantized",
    Mixed => "mixed",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingError {
    Empty,
    Zero,
    UnknownLabel,
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("embedding metadata text cannot be empty"),
            Self::Zero => formatter.write_str("embedding dimension must be positive"),
            Self::UnknownLabel => formatter.write_str("unknown embedding metadata label"),
        }
    }
}

impl Error for EmbeddingError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, EmbeddingError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(EmbeddingError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, EmbeddingError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(EmbeddingError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EmbeddingDimension, EmbeddingDistanceMetric, EmbeddingError, EmbeddingIndexKind,
        EmbeddingModelName, EmbeddingNormalizationKind, EmbeddingVectorShape,
    };

    #[test]
    fn validates_embedding_names_and_dimensions() -> Result<(), EmbeddingError> {
        let model = EmbeddingModelName::new(" text-embedding ")?;
        let dimension = EmbeddingDimension::new(384)?;
        let shape = EmbeddingVectorShape::new(dimension);

        assert_eq!(model.as_str(), "text-embedding");
        assert_eq!(shape.dimension().get(), 384);
        assert_eq!(EmbeddingDimension::new(0), Err(EmbeddingError::Zero));
        Ok(())
    }

    #[test]
    fn displays_and_parses_embedding_enums() -> Result<(), EmbeddingError> {
        assert_eq!(
            "dot product".parse::<EmbeddingDistanceMetric>()?,
            EmbeddingDistanceMetric::DotProduct
        );
        assert_eq!(
            "mean centered".parse::<EmbeddingNormalizationKind>()?,
            EmbeddingNormalizationKind::MeanCentered
        );
        assert_eq!(
            "ivf pq".parse::<EmbeddingIndexKind>()?,
            EmbeddingIndexKind::IvfPq
        );
        Ok(())
    }
}
