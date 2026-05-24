#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlModelArchitectureKind, MlModelArtifactKind, MlModelError, MlModelFormat, MlModelId,
        MlModelKind, MlModelLicense, MlModelName, MlModelProvider, MlModelStage, MlModelTask,
        MlModelVersion,
    };
}

macro_rules! model_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlModelError> {
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
            type Err = MlModelError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlModelError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! model_enum {
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
            type Err = MlModelError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlModelError::UnknownLabel),
                }
            }
        }
    };
}

model_text_newtype!(MlModelName);
model_text_newtype!(MlModelId);
model_text_newtype!(MlModelVersion);
model_text_newtype!(MlModelProvider);
model_text_newtype!(MlModelLicense);

model_enum!(MlModelKind {
    Classical => "classical",
    Linear => "linear",
    TreeBased => "tree-based",
    Kernel => "kernel",
    NeuralNetwork => "neural-network",
    Transformer => "transformer",
    Diffusion => "diffusion",
    Ensemble => "ensemble",
    RuleBased => "rule-based",
    Hybrid => "hybrid",
    Other => "other",
});

model_enum!(MlModelTask {
    Classification => "classification",
    Regression => "regression",
    Clustering => "clustering",
    Ranking => "ranking",
    Recommendation => "recommendation",
    Forecasting => "forecasting",
    Detection => "detection",
    Segmentation => "segmentation",
    Generation => "generation",
    Embedding => "embedding",
    AnomalyDetection => "anomaly-detection",
    Other => "other",
});

model_enum!(MlModelArchitectureKind {
    LinearModel => "linear-model",
    DecisionTree => "decision-tree",
    RandomForest => "random-forest",
    GradientBoostedTree => "gradient-boosted-tree",
    Svm => "svm",
    Knn => "knn",
    Mlp => "mlp",
    Cnn => "cnn",
    Rnn => "rnn",
    Lstm => "lstm",
    Transformer => "transformer",
    Autoencoder => "autoencoder",
    Diffusion => "diffusion",
    Other => "other",
});

model_enum!(MlModelArtifactKind {
    Weights => "weights",
    Config => "config",
    Tokenizer => "tokenizer",
    Vocabulary => "vocabulary",
    Preprocessor => "preprocessor",
    Pipeline => "pipeline",
    Checkpoint => "checkpoint",
    Bundle => "bundle",
    Card => "card",
    Metrics => "metrics",
    Other => "other",
});

model_enum!(MlModelFormat {
    Onnx => "onnx",
    Safetensors => "safetensors",
    Pickle => "pickle",
    Joblib => "joblib",
    TorchScript => "torch-script",
    TensorFlowSavedModel => "tensorflow-saved-model",
    CoreMl => "core-ml",
    Tflite => "tflite",
    OpenVino => "open-vino",
    Pmml => "pmml",
    Custom => "custom",
});

model_enum!(MlModelStage {
    Experimental => "experimental",
    Development => "development",
    Staging => "staging",
    Production => "production",
    Archived => "archived",
    Deprecated => "deprecated",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlModelError {
    Empty,
    UnknownLabel,
}

impl fmt::Display for MlModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML model metadata text cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown ML model metadata label"),
        }
    }
}

impl Error for MlModelError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlModelError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlModelError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlModelError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlModelError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlModelArchitectureKind, MlModelArtifactKind, MlModelError, MlModelFormat, MlModelKind,
        MlModelName, MlModelStage, MlModelTask,
    };

    #[test]
    fn validates_model_names() -> Result<(), MlModelError> {
        let name = MlModelName::new(" baseline ")?;

        assert_eq!(name.as_str(), "baseline");
        assert_eq!(name.to_string(), "baseline");
        assert_eq!("baseline".parse::<MlModelName>()?, name);
        Ok(())
    }

    #[test]
    fn rejects_empty_model_names() {
        assert_eq!(MlModelName::new("  "), Err(MlModelError::Empty));
    }

    #[test]
    fn displays_and_parses_model_enums() -> Result<(), MlModelError> {
        assert_eq!("tree based".parse::<MlModelKind>()?, MlModelKind::TreeBased);
        assert_eq!(
            "anomaly_detection".parse::<MlModelTask>()?,
            MlModelTask::AnomalyDetection
        );
        assert_eq!(
            "random forest".parse::<MlModelArchitectureKind>()?,
            MlModelArchitectureKind::RandomForest
        );
        assert_eq!(
            "checkpoint".parse::<MlModelArtifactKind>()?,
            MlModelArtifactKind::Checkpoint
        );
        assert_eq!(
            "torch_script".parse::<MlModelFormat>()?,
            MlModelFormat::TorchScript
        );
        assert_eq!(MlModelStage::Production.to_string(), "production");
        Ok(())
    }
}
