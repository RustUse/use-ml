#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlBatchingKind, MlConfidenceScore, MlInferenceError, MlInferenceMode, MlInferenceRequestId,
        MlInferenceStatus, MlInputKind, MlLatencyBucket, MlOutputKind, MlPredictionId,
        MlServingEndpointName, MlServingKind,
    };
}

macro_rules! inference_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlInferenceError> {
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
            type Err = MlInferenceError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlInferenceError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! inference_enum {
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
            type Err = MlInferenceError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlInferenceError::UnknownLabel),
                }
            }
        }
    };
}

inference_text_newtype!(MlInferenceRequestId);
inference_text_newtype!(MlPredictionId);
inference_text_newtype!(MlServingEndpointName);

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MlConfidenceScore(f64);

impl MlConfidenceScore {
    pub fn new(value: f64) -> Result<Self, MlInferenceError> {
        if !value.is_finite() {
            return Err(MlInferenceError::NonFinite);
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(MlInferenceError::OutOfRange);
        }
        Ok(Self(value))
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

inference_enum!(MlInferenceMode {
    Online => "online",
    Batch => "batch",
    Streaming => "streaming",
    Edge => "edge",
    Offline => "offline",
});

inference_enum!(MlInferenceStatus {
    Pending => "pending",
    Running => "running",
    Succeeded => "succeeded",
    Failed => "failed",
    Cancelled => "cancelled",
    TimedOut => "timed-out",
});

inference_enum!(MlServingKind {
    Local => "local",
    Embedded => "embedded",
    Api => "api",
    BatchJob => "batch-job",
    StreamProcessor => "stream-processor",
    EdgeDevice => "edge-device",
    Browser => "browser",
    Mobile => "mobile",
    Other => "other",
});

inference_enum!(MlInputKind {
    Text => "text",
    Image => "image",
    Audio => "audio",
    Video => "video",
    Tabular => "tabular",
    Json => "json",
    Tensor => "tensor",
    Embedding => "embedding",
    Multimodal => "multimodal",
    Other => "other",
});

inference_enum!(MlOutputKind {
    Class => "class",
    Score => "score",
    Ranking => "ranking",
    Text => "text",
    Image => "image",
    Audio => "audio",
    BoundingBox => "bounding-box",
    Mask => "mask",
    Embedding => "embedding",
    Tensor => "tensor",
    Json => "json",
    Other => "other",
});

inference_enum!(MlBatchingKind {
    None => "none",
    Fixed => "fixed",
    Dynamic => "dynamic",
    MicroBatch => "micro-batch",
    Adaptive => "adaptive",
});

inference_enum!(MlLatencyBucket {
    Sub10Ms => "sub-10-ms",
    Sub50Ms => "sub-50-ms",
    Sub100Ms => "sub-100-ms",
    Sub500Ms => "sub-500-ms",
    Sub1s => "sub-1s",
    Sub5s => "sub-5s",
    Over5s => "over-5s",
    Unknown => "unknown",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlInferenceError {
    Empty,
    NonFinite,
    OutOfRange,
    UnknownLabel,
}

impl fmt::Display for MlInferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML inference metadata text cannot be empty"),
            Self::NonFinite => formatter.write_str("ML confidence score must be finite"),
            Self::OutOfRange => formatter.write_str("ML confidence score must be in 0.0..=1.0"),
            Self::UnknownLabel => formatter.write_str("unknown ML inference metadata label"),
        }
    }
}

impl Error for MlInferenceError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlInferenceError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlInferenceError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlInferenceError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlInferenceError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlBatchingKind, MlConfidenceScore, MlInferenceError, MlInferenceMode, MlInferenceRequestId,
        MlInferenceStatus, MlInputKind, MlLatencyBucket, MlOutputKind, MlServingKind,
    };

    #[test]
    fn validates_inference_request_ids() -> Result<(), MlInferenceError> {
        let request = MlInferenceRequestId::new(" req-001 ")?;

        assert_eq!(request.as_str(), "req-001");
        assert_eq!("req-001".parse::<MlInferenceRequestId>()?, request);
        Ok(())
    }

    #[test]
    fn validates_confidence_scores() -> Result<(), MlInferenceError> {
        assert_eq!(MlConfidenceScore::new(0.0)?.value(), 0.0);
        assert_eq!(MlConfidenceScore::new(1.0)?.value(), 1.0);
        assert_eq!(
            MlConfidenceScore::new(-0.1),
            Err(MlInferenceError::OutOfRange)
        );
        assert_eq!(
            MlConfidenceScore::new(1.1),
            Err(MlInferenceError::OutOfRange)
        );
        assert_eq!(
            MlConfidenceScore::new(f64::INFINITY),
            Err(MlInferenceError::NonFinite)
        );
        Ok(())
    }

    #[test]
    fn displays_and_parses_inference_enums() -> Result<(), MlInferenceError> {
        assert_eq!(
            "online".parse::<MlInferenceMode>()?,
            MlInferenceMode::Online
        );
        assert_eq!(
            "timed out".parse::<MlInferenceStatus>()?,
            MlInferenceStatus::TimedOut
        );
        assert_eq!(
            "batch job".parse::<MlServingKind>()?,
            MlServingKind::BatchJob
        );
        assert_eq!("json".parse::<MlInputKind>()?, MlInputKind::Json);
        assert_eq!(
            "bounding box".parse::<MlOutputKind>()?,
            MlOutputKind::BoundingBox
        );
        assert_eq!(
            "micro_batch".parse::<MlBatchingKind>()?,
            MlBatchingKind::MicroBatch
        );
        assert_eq!(MlLatencyBucket::Sub100Ms.to_string(), "sub-100-ms");
        Ok(())
    }
}
