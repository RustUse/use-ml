#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlAnnotationKind, MlClassId, MlClassName, MlLabelCardinality, MlLabelError, MlLabelId,
        MlLabelKind, MlLabelName, MlLabelQuality, MlLabelSource, MlTargetKind,
    };
}

macro_rules! label_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlLabelError> {
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
            type Err = MlLabelError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlLabelError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! label_enum {
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
            type Err = MlLabelError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlLabelError::UnknownLabel),
                }
            }
        }
    };
}

label_text_newtype!(MlLabelName);
label_text_newtype!(MlLabelId);
label_text_newtype!(MlClassName);
label_text_newtype!(MlClassId);

label_enum!(MlLabelKind {
    Class => "class",
    Multiclass => "multiclass",
    Multilabel => "multilabel",
    RegressionTarget => "regression-target",
    RankingTarget => "ranking-target",
    SequenceTag => "sequence-tag",
    BoundingBox => "bounding-box",
    Mask => "mask",
    Span => "span",
    Other => "other",
});

label_enum!(MlTargetKind {
    BinaryClassification => "binary-classification",
    MulticlassClassification => "multiclass-classification",
    MultilabelClassification => "multilabel-classification",
    Regression => "regression",
    Ranking => "ranking",
    Forecasting => "forecasting",
    Clustering => "clustering",
    Generation => "generation",
    Other => "other",
});

label_enum!(MlAnnotationKind {
    Human => "human",
    Machine => "machine",
    Weak => "weak",
    Programmatic => "programmatic",
    Heuristic => "heuristic",
    Consensus => "consensus",
    Gold => "gold",
    Unknown => "unknown",
});

label_enum!(MlLabelSource {
    HumanAnnotator => "human-annotator",
    ExpertAnnotator => "expert-annotator",
    Model => "model",
    Rule => "rule",
    ExistingSystem => "existing-system",
    Synthetic => "synthetic",
    Unknown => "unknown",
});

label_enum!(MlLabelQuality {
    Unknown => "unknown",
    Low => "low",
    Medium => "medium",
    High => "high",
    Gold => "gold",
});

label_enum!(MlLabelCardinality {
    Single => "single",
    Multiple => "multiple",
    Sequence => "sequence",
    Dense => "dense",
    Sparse => "sparse",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlLabelError {
    Empty,
    UnknownLabel,
}

impl fmt::Display for MlLabelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML label metadata text cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown ML label metadata label"),
        }
    }
}

impl Error for MlLabelError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlLabelError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlLabelError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlLabelError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlLabelError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlAnnotationKind, MlClassName, MlLabelCardinality, MlLabelError, MlLabelKind, MlLabelName,
        MlLabelQuality, MlLabelSource, MlTargetKind,
    };

    #[test]
    fn validates_label_and_class_names() -> Result<(), MlLabelError> {
        let label = MlLabelName::new(" species ")?;
        let class = MlClassName::new("setosa")?;

        assert_eq!(label.as_str(), "species");
        assert_eq!(class.as_str(), "setosa");
        assert_eq!("species".parse::<MlLabelName>()?, label);
        Ok(())
    }

    #[test]
    fn rejects_empty_label_and_class_names() {
        assert_eq!(MlLabelName::new("  "), Err(MlLabelError::Empty));
        assert_eq!(MlClassName::new("\t"), Err(MlLabelError::Empty));
    }

    #[test]
    fn displays_and_parses_label_enums() -> Result<(), MlLabelError> {
        assert_eq!(
            "bounding box".parse::<MlLabelKind>()?,
            MlLabelKind::BoundingBox
        );
        assert_eq!(
            "binary_classification".parse::<MlTargetKind>()?,
            MlTargetKind::BinaryClassification
        );
        assert_eq!(
            "human".parse::<MlAnnotationKind>()?,
            MlAnnotationKind::Human
        );
        assert_eq!(
            "expert annotator".parse::<MlLabelSource>()?,
            MlLabelSource::ExpertAnnotator
        );
        assert_eq!("gold".parse::<MlLabelQuality>()?, MlLabelQuality::Gold);
        assert_eq!(
            "sparse".parse::<MlLabelCardinality>()?,
            MlLabelCardinality::Sparse
        );
        Ok(())
    }
}
