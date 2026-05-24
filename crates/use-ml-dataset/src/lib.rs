#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlDatasetCardRef, MlDatasetError, MlDatasetId, MlDatasetKind, MlDatasetLicense,
        MlDatasetName, MlDatasetProvenance, MlDatasetSchemaRef, MlDatasetSplit, MlDatasetVersion,
        MlExampleId, MlExampleKind,
    };
}

macro_rules! dataset_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlDatasetError> {
                non_empty_text(value).map(Self)
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
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
            type Err = MlDatasetError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlDatasetError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! dataset_enum {
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
            type Err = MlDatasetError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlDatasetError::UnknownLabel),
                }
            }
        }
    };
}

dataset_text_newtype!(MlDatasetName);
dataset_text_newtype!(MlDatasetId);
dataset_text_newtype!(MlDatasetVersion);
dataset_text_newtype!(MlExampleId);
dataset_text_newtype!(MlDatasetLicense);
dataset_text_newtype!(MlDatasetSchemaRef);
dataset_text_newtype!(MlDatasetCardRef);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MlDatasetSplit {
    Train,
    Validation,
    Test,
    Holdout,
    Calibration,
    Shadow,
    Production,
    Custom(String),
}

impl fmt::Display for MlDatasetSplit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Train => formatter.write_str("train"),
            Self::Validation => formatter.write_str("validation"),
            Self::Test => formatter.write_str("test"),
            Self::Holdout => formatter.write_str("holdout"),
            Self::Calibration => formatter.write_str("calibration"),
            Self::Shadow => formatter.write_str("shadow"),
            Self::Production => formatter.write_str("production"),
            Self::Custom(value) => formatter.write_str(value),
        }
    }
}

impl FromStr for MlDatasetSplit {
    type Err = MlDatasetError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            return Err(MlDatasetError::Empty);
        }

        Ok(match normalized_label(trimmed)?.as_str() {
            "train" | "training" => Self::Train,
            "validation" | "valid" | "val" => Self::Validation,
            "test" => Self::Test,
            "holdout" => Self::Holdout,
            "calibration" => Self::Calibration,
            "shadow" => Self::Shadow,
            "production" | "prod" => Self::Production,
            _ => Self::Custom(trimmed.to_string()),
        })
    }
}

dataset_enum!(MlDatasetKind {
    Tabular => "tabular",
    Text => "text",
    Image => "image",
    Audio => "audio",
    Video => "video",
    TimeSeries => "time-series",
    Graph => "graph",
    Multimodal => "multimodal",
    Synthetic => "synthetic",
    Other => "other",
});

dataset_enum!(MlExampleKind {
    Labeled => "labeled",
    Unlabeled => "unlabeled",
    WeaklyLabeled => "weakly-labeled",
    PseudoLabeled => "pseudo-labeled",
    Augmented => "augmented",
});

dataset_enum!(MlDatasetProvenance {
    HumanCreated => "human-created",
    MachineGenerated => "machine-generated",
    Synthetic => "synthetic",
    Scraped => "scraped",
    Instrumented => "instrumented",
    Mixed => "mixed",
    Unknown => "unknown",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlDatasetError {
    Empty,
    UnknownLabel,
}

impl fmt::Display for MlDatasetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML dataset metadata text cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown ML dataset metadata label"),
        }
    }
}

impl Error for MlDatasetError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlDatasetError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlDatasetError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlDatasetError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlDatasetError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlDatasetError, MlDatasetKind, MlDatasetName, MlDatasetProvenance, MlDatasetSplit,
        MlExampleKind,
    };

    #[test]
    fn validates_dataset_names() -> Result<(), MlDatasetError> {
        let name = MlDatasetName::new(" iris ")?;

        assert_eq!(name.as_str(), "iris");
        assert_eq!(name.to_string(), "iris");
        assert_eq!("iris".parse::<MlDatasetName>()?, name);
        Ok(())
    }

    #[test]
    fn rejects_empty_dataset_names() {
        assert_eq!(MlDatasetName::new("  "), Err(MlDatasetError::Empty));
    }

    #[test]
    fn displays_and_parses_dataset_enums() -> Result<(), MlDatasetError> {
        assert_eq!(MlDatasetSplit::Validation.to_string(), "validation");
        assert_eq!(
            "time_series".parse::<MlDatasetKind>()?,
            MlDatasetKind::TimeSeries
        );
        assert_eq!(
            "pseudo labeled".parse::<MlExampleKind>()?,
            MlExampleKind::PseudoLabeled
        );
        assert_eq!(
            "machine-generated".parse::<MlDatasetProvenance>()?,
            MlDatasetProvenance::MachineGenerated
        );
        assert_eq!(
            "shadow-2026".parse::<MlDatasetSplit>()?,
            MlDatasetSplit::Custom("shadow-2026".to_string())
        );
        Ok(())
    }
}
