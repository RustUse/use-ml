#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::{error::Error, num::NonZeroUsize};

pub mod prelude {
    pub use crate::{
        MlBatchSize, MlCheckpointKind, MlEpochCount, MlHyperparameterName, MlHyperparameterValue,
        MlLearningRate, MlLossKind, MlOptimizerKind, MlTrainingError, MlTrainingJobName,
        MlTrainingPhase, MlTrainingRunId, MlTrainingStatus,
    };
}

macro_rules! training_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlTrainingError> {
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
            type Err = MlTrainingError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlTrainingError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! training_enum {
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
            type Err = MlTrainingError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlTrainingError::UnknownLabel),
                }
            }
        }
    };
}

training_text_newtype!(MlTrainingRunId);
training_text_newtype!(MlTrainingJobName);
training_text_newtype!(MlHyperparameterName);
training_text_newtype!(MlHyperparameterValue);

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MlBatchSize(NonZeroUsize);

impl MlBatchSize {
    pub fn new(value: usize) -> Result<Self, MlTrainingError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(MlTrainingError::Zero)
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MlEpochCount(NonZeroUsize);

impl MlEpochCount {
    pub fn new(value: usize) -> Result<Self, MlTrainingError> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or(MlTrainingError::Zero)
    }

    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MlLearningRate(f64);

impl MlLearningRate {
    pub fn new(value: f64) -> Result<Self, MlTrainingError> {
        if !value.is_finite() {
            return Err(MlTrainingError::NonFinite);
        }
        if value <= 0.0 {
            return Err(MlTrainingError::NonPositive);
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

training_enum!(MlTrainingStatus {
    Queued => "queued",
    Running => "running",
    Succeeded => "succeeded",
    Failed => "failed",
    Cancelled => "cancelled",
    TimedOut => "timed-out",
    Paused => "paused",
    Unknown => "unknown",
});

training_enum!(MlTrainingPhase {
    PrepareData => "prepare-data",
    Initialize => "initialize",
    Train => "train",
    Validate => "validate",
    Tune => "tune",
    Checkpoint => "checkpoint",
    Evaluate => "evaluate",
    Export => "export",
    Complete => "complete",
});

training_enum!(MlOptimizerKind {
    Sgd => "sgd",
    Momentum => "momentum",
    Adam => "adam",
    AdamW => "adamw",
    RmsProp => "rmsprop",
    Adagrad => "adagrad",
    Adadelta => "adadelta",
    Lbfgs => "lbfgs",
    Custom => "custom",
});

training_enum!(MlLossKind {
    CrossEntropy => "cross-entropy",
    BinaryCrossEntropy => "binary-cross-entropy",
    MeanSquaredError => "mean-squared-error",
    MeanAbsoluteError => "mean-absolute-error",
    Huber => "huber",
    Hinge => "hinge",
    Triplet => "triplet",
    Contrastive => "contrastive",
    Custom => "custom",
});

training_enum!(MlCheckpointKind {
    Best => "best",
    Latest => "latest",
    Epoch => "epoch",
    Step => "step",
    Manual => "manual",
    Final => "final",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlTrainingError {
    Empty,
    Zero,
    NonFinite,
    NonPositive,
    UnknownLabel,
}

impl fmt::Display for MlTrainingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML training metadata text cannot be empty"),
            Self::Zero => formatter.write_str("ML training count must be positive"),
            Self::NonFinite => formatter.write_str("ML training value must be finite"),
            Self::NonPositive => formatter.write_str("ML training value must be positive"),
            Self::UnknownLabel => formatter.write_str("unknown ML training metadata label"),
        }
    }
}

impl Error for MlTrainingError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlTrainingError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlTrainingError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlTrainingError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlTrainingError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlBatchSize, MlCheckpointKind, MlEpochCount, MlLearningRate, MlLossKind, MlOptimizerKind,
        MlTrainingError, MlTrainingRunId, MlTrainingStatus,
    };

    #[test]
    fn validates_training_ids() -> Result<(), MlTrainingError> {
        let run_id = MlTrainingRunId::new(" run-001 ")?;

        assert_eq!(run_id.as_str(), "run-001");
        assert_eq!("run-001".parse::<MlTrainingRunId>()?, run_id);
        Ok(())
    }

    #[test]
    fn validates_positive_counts_and_learning_rates() -> Result<(), MlTrainingError> {
        assert_eq!(MlBatchSize::new(32)?.get(), 32);
        assert_eq!(MlEpochCount::new(10)?.get(), 10);
        assert_eq!(MlLearningRate::new(0.001)?.value(), 0.001);
        assert_eq!(MlBatchSize::new(0), Err(MlTrainingError::Zero));
        assert_eq!(MlEpochCount::new(0), Err(MlTrainingError::Zero));
        assert_eq!(MlLearningRate::new(0.0), Err(MlTrainingError::NonPositive));
        assert_eq!(
            MlLearningRate::new(f64::NAN),
            Err(MlTrainingError::NonFinite)
        );
        Ok(())
    }

    #[test]
    fn displays_and_parses_training_enums() -> Result<(), MlTrainingError> {
        assert_eq!(
            "timed out".parse::<MlTrainingStatus>()?,
            MlTrainingStatus::TimedOut
        );
        assert_eq!("adamw".parse::<MlOptimizerKind>()?, MlOptimizerKind::AdamW);
        assert_eq!(
            "mean squared error".parse::<MlLossKind>()?,
            MlLossKind::MeanSquaredError
        );
        assert_eq!(MlCheckpointKind::Latest.to_string(), "latest");
        Ok(())
    }
}
