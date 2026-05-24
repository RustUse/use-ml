#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlFeatureDriftStatus, MlFeatureEncodingKind, MlFeatureError, MlFeatureId, MlFeatureKind,
        MlFeatureMissingValuePolicy, MlFeatureName, MlFeatureRole, MlFeatureScalingKind,
        MlFeatureSource, MlFeatureTransformKind,
    };
}

macro_rules! feature_text_newtype {
    ($name:ident, ascii) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlFeatureError> {
                ascii_safe_text(value).map(Self)
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
            type Err = MlFeatureError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlFeatureError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
    ($name:ident, text) => {
        #[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl AsRef<str>) -> Result<Self, MlFeatureError> {
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
            type Err = MlFeatureError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                Self::new(value)
            }
        }

        impl TryFrom<&str> for $name {
            type Error = MlFeatureError;

            fn try_from(value: &str) -> Result<Self, Self::Error> {
                Self::new(value)
            }
        }
    };
}

macro_rules! feature_enum {
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
            type Err = MlFeatureError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlFeatureError::UnknownLabel),
                }
            }
        }
    };
}

feature_text_newtype!(MlFeatureName, ascii);
feature_text_newtype!(MlFeatureId, text);

feature_enum!(MlFeatureKind {
    Numeric => "numeric",
    Categorical => "categorical",
    Ordinal => "ordinal",
    Boolean => "boolean",
    Text => "text",
    Image => "image",
    Audio => "audio",
    Video => "video",
    Timestamp => "timestamp",
    Geospatial => "geospatial",
    Vector => "vector",
    Embedding => "embedding",
    Graph => "graph",
    Other => "other",
});

feature_enum!(MlFeatureRole {
    Input => "input",
    Target => "target",
    Weight => "weight",
    Group => "group",
    Timestamp => "timestamp",
    Identifier => "identifier",
    Metadata => "metadata",
    Ignore => "ignore",
});

feature_enum!(MlFeatureSource {
    Raw => "raw",
    Derived => "derived",
    Aggregated => "aggregated",
    Joined => "joined",
    UserProvided => "user-provided",
    SystemGenerated => "system-generated",
    External => "external",
    Synthetic => "synthetic",
});

feature_enum!(MlFeatureTransformKind {
    Normalize => "normalize",
    Standardize => "standardize",
    Bucketize => "bucketize",
    OneHotEncode => "one-hot-encode",
    Tokenize => "tokenize",
    Embed => "embed",
    Impute => "impute",
    Clip => "clip",
    Log => "log",
    Custom => "custom",
});

feature_enum!(MlFeatureEncodingKind {
    None => "none",
    OneHot => "one-hot",
    Ordinal => "ordinal",
    Label => "label",
    Binary => "binary",
    Hashing => "hashing",
    Token => "token",
    Embedding => "embedding",
    Custom => "custom",
});

feature_enum!(MlFeatureScalingKind {
    None => "none",
    MinMax => "min-max",
    Standard => "standard",
    Robust => "robust",
    UnitNorm => "unit-norm",
    Log => "log",
    Custom => "custom",
});

feature_enum!(MlFeatureMissingValuePolicy {
    Allow => "allow",
    Drop => "drop",
    ImputeMean => "impute-mean",
    ImputeMedian => "impute-median",
    ImputeMode => "impute-mode",
    ImputeConstant => "impute-constant",
    Unknown => "unknown",
});

feature_enum!(MlFeatureDriftStatus {
    Unknown => "unknown",
    Stable => "stable",
    Warning => "warning",
    Drifted => "drifted",
});

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlFeatureError {
    Empty,
    NonAsciiSafeName,
    UnknownLabel,
}

impl fmt::Display for MlFeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML feature metadata text cannot be empty"),
            Self::NonAsciiSafeName => formatter.write_str("ML feature name must be ASCII-safe"),
            Self::UnknownLabel => formatter.write_str("unknown ML feature metadata label"),
        }
    }
}

impl Error for MlFeatureError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlFeatureError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlFeatureError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn ascii_safe_text(value: impl AsRef<str>) -> Result<String, MlFeatureError> {
    let trimmed = non_empty_text(value)?;
    if trimmed
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b'/'))
    {
        Ok(trimmed)
    } else {
        Err(MlFeatureError::NonAsciiSafeName)
    }
}

fn normalized_label(value: &str) -> Result<String, MlFeatureError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlFeatureError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlFeatureDriftStatus, MlFeatureEncodingKind, MlFeatureError, MlFeatureKind,
        MlFeatureMissingValuePolicy, MlFeatureName, MlFeatureRole, MlFeatureScalingKind,
        MlFeatureTransformKind,
    };

    #[test]
    fn validates_ascii_safe_feature_names() -> Result<(), MlFeatureError> {
        let name = MlFeatureName::new(" sepal_width ")?;

        assert_eq!(name.as_str(), "sepal_width");
        assert_eq!(name.to_string(), "sepal_width");
        assert_eq!("sepal_width".parse::<MlFeatureName>()?, name);
        Ok(())
    }

    #[test]
    fn rejects_invalid_feature_names() {
        assert_eq!(MlFeatureName::new("  "), Err(MlFeatureError::Empty));
        assert_eq!(
            MlFeatureName::new("prompt variable"),
            Err(MlFeatureError::NonAsciiSafeName)
        );
        assert_eq!(
            MlFeatureName::new("城市"),
            Err(MlFeatureError::NonAsciiSafeName)
        );
    }

    #[test]
    fn displays_and_parses_feature_enums() -> Result<(), MlFeatureError> {
        assert_eq!(
            "one_hot".parse::<MlFeatureEncodingKind>()?,
            MlFeatureEncodingKind::OneHot
        );
        assert_eq!(
            "unit norm".parse::<MlFeatureScalingKind>()?,
            MlFeatureScalingKind::UnitNorm
        );
        assert_eq!(
            "timestamp".parse::<MlFeatureRole>()?,
            MlFeatureRole::Timestamp
        );
        assert_eq!("numeric".parse::<MlFeatureKind>()?, MlFeatureKind::Numeric);
        assert_eq!(
            "drifted".parse::<MlFeatureDriftStatus>()?,
            MlFeatureDriftStatus::Drifted
        );
        assert_eq!(
            "impute mean".parse::<MlFeatureMissingValuePolicy>()?,
            MlFeatureMissingValuePolicy::ImputeMean
        );
        assert_eq!(
            MlFeatureTransformKind::OneHotEncode.to_string(),
            "one-hot-encode"
        );
        Ok(())
    }
}
