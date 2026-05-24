#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlPipelineArtifactKind, MlPipelineDependencyKind, MlPipelineError, MlPipelineId,
        MlPipelineName, MlPipelineRunId, MlPipelineScheduleKind, MlPipelineStatus,
        MlPipelineStepKind, MlPipelineStepName, MlPipelineTriggerKind,
    };
}

macro_rules! pipeline_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlPipelineError> {
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
            type Err = MlPipelineError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlPipelineError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! pipeline_enum {
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
            type Err = MlPipelineError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlPipelineError::UnknownLabel),
                }
            }
        }
    };
}

pipeline_text_newtype!(MlPipelineName);
pipeline_text_newtype!(MlPipelineId);
pipeline_text_newtype!(MlPipelineStepName);
pipeline_text_newtype!(MlPipelineRunId);

pipeline_enum!(MlPipelineStepKind {
    Ingest => "ingest",
    Validate => "validate",
    Clean => "clean",
    Transform => "transform",
    Featurize => "featurize",
    Split => "split",
    Train => "train",
    Tune => "tune",
    Evaluate => "evaluate",
    Register => "register",
    Deploy => "deploy",
    Monitor => "monitor",
    Rollback => "rollback",
    Other => "other",
});

pipeline_enum!(MlPipelineStatus {
    Draft => "draft",
    Ready => "ready",
    Running => "running",
    Succeeded => "succeeded",
    Failed => "failed",
    Cancelled => "cancelled",
    Paused => "paused",
    Deprecated => "deprecated",
});

pipeline_enum!(MlPipelineArtifactKind {
    Dataset => "dataset",
    FeatureSet => "feature-set",
    Model => "model",
    Metrics => "metrics",
    Report => "report",
    Config => "config",
    Checkpoint => "checkpoint",
    Prediction => "prediction",
    Log => "log",
    Other => "other",
});

pipeline_enum!(MlPipelineDependencyKind {
    Data => "data",
    Model => "model",
    Config => "config",
    Secret => "secret",
    Service => "service",
    Compute => "compute",
    HumanApproval => "human-approval",
    Other => "other",
});

pipeline_enum!(MlPipelineTriggerKind {
    Manual => "manual",
    Schedule => "schedule",
    Commit => "commit",
    DataArrival => "data-arrival",
    ModelChange => "model-change",
    DriftDetected => "drift-detected",
    Api => "api",
    Other => "other",
});

pipeline_enum!(MlPipelineScheduleKind {
    None => "none",
    Once => "once",
    Hourly => "hourly",
    Daily => "daily",
    Weekly => "weekly",
    Monthly => "monthly",
    Cron => "cron",
    EventDriven => "event-driven",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlPipelineError {
    Empty,
    UnknownLabel,
}

impl fmt::Display for MlPipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML pipeline metadata text cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown ML pipeline metadata label"),
        }
    }
}

impl Error for MlPipelineError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlPipelineError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlPipelineError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlPipelineError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlPipelineError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlPipelineError, MlPipelineName, MlPipelineScheduleKind, MlPipelineStatus,
        MlPipelineStepKind,
    };

    #[test]
    fn validates_pipeline_names() -> Result<(), MlPipelineError> {
        let name = MlPipelineName::new(" training ")?;

        assert_eq!(name.as_str(), "training");
        assert_eq!("training".parse::<MlPipelineName>()?, name);
        assert_eq!(MlPipelineName::new("  "), Err(MlPipelineError::Empty));
        Ok(())
    }

    #[test]
    fn displays_and_parses_pipeline_enums() -> Result<(), MlPipelineError> {
        assert_eq!(
            "featurize".parse::<MlPipelineStepKind>()?,
            MlPipelineStepKind::Featurize
        );
        assert_eq!(
            "ready".parse::<MlPipelineStatus>()?,
            MlPipelineStatus::Ready
        );
        assert_eq!(
            "event driven".parse::<MlPipelineScheduleKind>()?,
            MlPipelineScheduleKind::EventDriven
        );
        Ok(())
    }
}
