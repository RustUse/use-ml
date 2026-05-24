#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlArtifactUri, MlExperimentError, MlExperimentId, MlExperimentName, MlExperimentStage,
        MlParameterName, MlParameterValue, MlRunId, MlRunStatus, MlRunTag, MlTrackingBackendKind,
    };
}

macro_rules! experiment_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlExperimentError> {
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
            type Err = MlExperimentError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlExperimentError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! experiment_enum {
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
            type Err = MlExperimentError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlExperimentError::UnknownLabel),
                }
            }
        }
    };
}

experiment_text_newtype!(MlExperimentName);
experiment_text_newtype!(MlExperimentId);
experiment_text_newtype!(MlRunId);
experiment_text_newtype!(MlRunTag);
experiment_text_newtype!(MlParameterName);
experiment_text_newtype!(MlParameterValue);
experiment_text_newtype!(MlArtifactUri);

experiment_enum!(MlRunStatus {
    Scheduled => "scheduled",
    Running => "running",
    Finished => "finished",
    Failed => "failed",
    Killed => "killed",
    Cancelled => "cancelled",
    Unknown => "unknown",
});

experiment_enum!(MlTrackingBackendKind {
    Local => "local",
    FileStore => "file-store",
    Sql => "sql",
    MlflowLike => "mlflow-like",
    WeightsAndBiasesLike => "weights-and-biases-like",
    NeptuneLike => "neptune-like",
    Custom => "custom",
});

experiment_enum!(MlExperimentStage {
    Exploration => "exploration",
    Baseline => "baseline",
    Tuning => "tuning",
    Candidate => "candidate",
    Validation => "validation",
    ProductionCandidate => "production-candidate",
    Archived => "archived",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlExperimentError {
    Empty,
    UnknownLabel,
}

impl fmt::Display for MlExperimentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML experiment metadata text cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown ML experiment metadata label"),
        }
    }
}

impl Error for MlExperimentError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlExperimentError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlExperimentError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlExperimentError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlExperimentError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlExperimentError, MlExperimentName, MlExperimentStage, MlRunId, MlRunStatus,
        MlTrackingBackendKind,
    };

    #[test]
    fn validates_experiment_names_and_run_ids() -> Result<(), MlExperimentError> {
        let experiment = MlExperimentName::new(" baseline ")?;
        let run = MlRunId::new("run-001")?;

        assert_eq!(experiment.as_str(), "baseline");
        assert_eq!(run.as_str(), "run-001");
        assert_eq!(MlExperimentName::new("  "), Err(MlExperimentError::Empty));
        Ok(())
    }

    #[test]
    fn displays_and_parses_experiment_enums() -> Result<(), MlExperimentError> {
        assert_eq!("finished".parse::<MlRunStatus>()?, MlRunStatus::Finished);
        assert_eq!(
            "weights and biases like".parse::<MlTrackingBackendKind>()?,
            MlTrackingBackendKind::WeightsAndBiasesLike
        );
        assert_eq!(
            "production candidate".parse::<MlExperimentStage>()?,
            MlExperimentStage::ProductionCandidate
        );
        Ok(())
    }
}
