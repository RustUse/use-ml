#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::{error::Error, num::NonZeroUsize};

pub mod prelude {
    pub use crate::{
        MlBenchmarkName, MlConfusionMatrixShape, MlEvalSliceKind, MlEvalSliceName,
        MlEvaluationError, MlEvaluationKind, MlEvaluationRunId, MlEvaluationStatus,
        MlEvaluationTarget, MlThreshold, MlValidationStrategy,
    };
}

macro_rules! evaluation_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlEvaluationError> {
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
            type Err = MlEvaluationError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlEvaluationError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! evaluation_enum {
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
            type Err = MlEvaluationError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlEvaluationError::UnknownLabel),
                }
            }
        }
    };
}

evaluation_text_newtype!(MlEvaluationRunId);
evaluation_text_newtype!(MlEvalSliceName);
evaluation_text_newtype!(MlBenchmarkName);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MlThreshold(f64);

impl MlThreshold {
    pub fn new(value: f64) -> Result<Self, MlEvaluationError> {
        if !value.is_finite() {
            return Err(MlEvaluationError::NonFinite);
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(MlEvaluationError::OutOfRange);
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MlConfusionMatrixShape {
    rows: NonZeroUsize,
    columns: NonZeroUsize,
}

impl MlConfusionMatrixShape {
    pub fn new(rows: usize, columns: usize) -> Result<Self, MlEvaluationError> {
        Ok(Self {
            rows: NonZeroUsize::new(rows).ok_or(MlEvaluationError::Zero)?,
            columns: NonZeroUsize::new(columns).ok_or(MlEvaluationError::Zero)?,
        })
    }

    pub const fn rows(self) -> usize {
        self.rows.get()
    }

    pub const fn columns(self) -> usize {
        self.columns.get()
    }

    pub const fn is_square(self) -> bool {
        self.rows.get() == self.columns.get()
    }
}

evaluation_enum!(MlEvaluationKind {
    Offline => "offline",
    Online => "online",
    Shadow => "shadow",
    ABTest => "a-b-test",
    Backtest => "backtest",
    CrossValidation => "cross-validation",
    Holdout => "holdout",
    Benchmark => "benchmark",
    HumanEval => "human-eval",
    Other => "other",
});

evaluation_enum!(MlValidationStrategy {
    Holdout => "holdout",
    KFold => "k-fold",
    StratifiedKFold => "stratified-k-fold",
    TimeSeriesSplit => "time-series-split",
    LeaveOneOut => "leave-one-out",
    Bootstrap => "bootstrap",
    Custom => "custom",
});

evaluation_enum!(MlEvaluationStatus {
    Pending => "pending",
    Running => "running",
    Succeeded => "succeeded",
    Failed => "failed",
    Cancelled => "cancelled",
    Inconclusive => "inconclusive",
});

evaluation_enum!(MlEvaluationTarget {
    Model => "model",
    Pipeline => "pipeline",
    Dataset => "dataset",
    Feature => "feature",
    Label => "label",
    Artifact => "artifact",
    TrainingRun => "training-run",
    Other => "other",
});

evaluation_enum!(MlEvalSliceKind {
    Global => "global",
    Class => "class",
    Segment => "segment",
    Cohort => "cohort",
    Geography => "geography",
    TimeWindow => "time-window",
    Device => "device",
    Language => "language",
    Custom => "custom",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlEvaluationError {
    Empty,
    NonFinite,
    OutOfRange,
    Zero,
    UnknownLabel,
}

impl fmt::Display for MlEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML evaluation metadata text cannot be empty"),
            Self::NonFinite => formatter.write_str("ML evaluation value must be finite"),
            Self::OutOfRange => formatter.write_str("ML evaluation threshold must be in 0.0..=1.0"),
            Self::Zero => formatter.write_str("ML evaluation count must be positive"),
            Self::UnknownLabel => formatter.write_str("unknown ML evaluation metadata label"),
        }
    }
}

impl Error for MlEvaluationError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlEvaluationError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlEvaluationError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlEvaluationError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlEvaluationError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlConfusionMatrixShape, MlEvaluationError, MlEvaluationKind, MlEvaluationRunId,
        MlEvaluationStatus, MlThreshold, MlValidationStrategy,
    };

    #[test]
    fn validates_evaluation_run_ids() -> Result<(), MlEvaluationError> {
        let run_id = MlEvaluationRunId::new(" eval-001 ")?;

        assert_eq!(run_id.as_str(), "eval-001");
        assert_eq!("eval-001".parse::<MlEvaluationRunId>()?, run_id);
        Ok(())
    }

    #[test]
    fn validates_thresholds_and_confusion_matrix_shapes() -> Result<(), MlEvaluationError> {
        assert_eq!(MlThreshold::new(0.0)?.value(), 0.0);
        assert_eq!(MlThreshold::new(1.0)?.value(), 1.0);
        assert_eq!(MlThreshold::new(-0.1), Err(MlEvaluationError::OutOfRange));
        assert_eq!(MlThreshold::new(1.1), Err(MlEvaluationError::OutOfRange));
        assert_eq!(
            MlThreshold::new(f64::NAN),
            Err(MlEvaluationError::NonFinite)
        );

        let shape = MlConfusionMatrixShape::new(3, 3)?;
        assert_eq!(shape.rows(), 3);
        assert!(shape.is_square());
        assert_eq!(
            MlConfusionMatrixShape::new(0, 3),
            Err(MlEvaluationError::Zero)
        );
        Ok(())
    }

    #[test]
    fn displays_and_parses_evaluation_enums() -> Result<(), MlEvaluationError> {
        assert_eq!(
            "a b test".parse::<MlEvaluationKind>()?,
            MlEvaluationKind::ABTest
        );
        assert_eq!(
            "stratified k fold".parse::<MlValidationStrategy>()?,
            MlValidationStrategy::StratifiedKFold
        );
        assert_eq!(
            "cancelled".parse::<MlEvaluationStatus>()?,
            MlEvaluationStatus::Cancelled
        );
        Ok(())
    }
}
