#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlModelCard, MlModelCardAudience, MlModelCardDatasetRef, MlModelCardError,
        MlModelCardEvaluationSummary, MlModelCardIntendedUse, MlModelCardLimitation,
        MlModelCardName, MlModelCardOwner, MlModelCardRisk, MlModelCardSection,
    };
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlModelCard {
    name: MlModelCardName,
    sections: Vec<MlModelCardSection>,
}

impl MlModelCard {
    pub fn new(name: MlModelCardName) -> Self {
        Self {
            name,
            sections: Vec::new(),
        }
    }

    pub fn name(&self) -> &MlModelCardName {
        &self.name
    }

    pub fn sections(&self) -> &[MlModelCardSection] {
        &self.sections
    }

    pub fn with_section(mut self, section: MlModelCardSection) -> Self {
        self.sections.push(section);
        self
    }
}

macro_rules! model_card_text_newtype {
    ($name:ident) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlModelCardError> {
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
            type Err = MlModelCardError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlModelCardError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! model_card_enum {
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
            type Err = MlModelCardError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlModelCardError::UnknownLabel),
                }
            }
        }
    };
}

model_card_text_newtype!(MlModelCardName);
model_card_text_newtype!(MlModelCardLimitation);
model_card_text_newtype!(MlModelCardEvaluationSummary);
model_card_text_newtype!(MlModelCardDatasetRef);
model_card_text_newtype!(MlModelCardOwner);

model_card_enum!(MlModelCardSection {
    Overview => "overview",
    IntendedUse => "intended-use",
    Factors => "factors",
    Metrics => "metrics",
    EvaluationData => "evaluation-data",
    TrainingData => "training-data",
    EthicalConsiderations => "ethical-considerations",
    CaveatsAndRecommendations => "caveats-and-recommendations",
    Limitations => "limitations",
    Contact => "contact",
    License => "license",
    Other => "other",
});

model_card_enum!(MlModelCardAudience {
    Developer => "developer",
    Researcher => "researcher",
    Operator => "operator",
    EndUser => "end-user",
    Auditor => "auditor",
    Regulator => "regulator",
    Public => "public",
    Internal => "internal",
});

model_card_enum!(MlModelCardIntendedUse {
    Research => "research",
    Production => "production",
    Education => "education",
    Evaluation => "evaluation",
    Demo => "demo",
    InternalTooling => "internal-tooling",
    Other => "other",
});

model_card_enum!(MlModelCardRisk {
    Bias => "bias",
    Privacy => "privacy",
    Security => "security",
    Safety => "safety",
    Misuse => "misuse",
    Performance => "performance",
    DataQuality => "data-quality",
    DistributionShift => "distribution-shift",
    Other => "other",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlModelCardError {
    Empty,
    UnknownLabel,
}

impl fmt::Display for MlModelCardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML model-card metadata text cannot be empty"),
            Self::UnknownLabel => formatter.write_str("unknown ML model-card metadata label"),
        }
    }
}

impl Error for MlModelCardError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlModelCardError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlModelCardError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlModelCardError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlModelCardError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlModelCard, MlModelCardAudience, MlModelCardError, MlModelCardName, MlModelCardRisk,
        MlModelCardSection,
    };

    #[test]
    fn validates_model_card_names_and_builds_cards() -> Result<(), MlModelCardError> {
        let card = MlModelCard::new(MlModelCardName::new(" baseline-card ")?)
            .with_section(MlModelCardSection::Overview);

        assert_eq!(card.name().as_str(), "baseline-card");
        assert_eq!(card.sections(), &[MlModelCardSection::Overview]);
        assert_eq!(MlModelCardName::new("  "), Err(MlModelCardError::Empty));
        Ok(())
    }

    #[test]
    fn displays_and_parses_model_card_enums() -> Result<(), MlModelCardError> {
        assert_eq!(
            "ethical considerations".parse::<MlModelCardSection>()?,
            MlModelCardSection::EthicalConsiderations
        );
        assert_eq!(
            "end user".parse::<MlModelCardAudience>()?,
            MlModelCardAudience::EndUser
        );
        assert_eq!(
            "data quality".parse::<MlModelCardRisk>()?,
            MlModelCardRisk::DataQuality
        );
        Ok(())
    }
}
