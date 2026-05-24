#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use core::{fmt, str::FromStr};
use std::error::Error;

pub mod prelude {
    pub use crate::{
        MlClassificationMetric, MlClusteringMetric, MlGenerationMetric, MlMetricAggregation,
        MlMetricDirection, MlMetricError, MlMetricKind, MlMetricName, MlMetricValue,
        MlRankingMetric, MlRegressionMetric,
    };
}

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MlMetricName(String);

impl MlMetricName {
    pub fn new(value: impl AsRef<str>) -> Result<Self, MlMetricError> {
        non_empty_text(value).map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for MlMetricName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for MlMetricName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for MlMetricName {
    type Err = MlMetricError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

impl TryFrom<&str> for MlMetricName {
    type Error = MlMetricError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct MlMetricValue(f64);

impl MlMetricValue {
    pub fn new(value: f64) -> Result<Self, MlMetricError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(MlMetricError::NonFinite)
        }
    }

    pub const fn value(self) -> f64 {
        self.0
    }
}

macro_rules! metric_enum {
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
            type Err = MlMetricError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                match normalized_label(value)?.as_str() {
                    $($label => Ok(Self::$variant),)+
                    _ => Err(MlMetricError::UnknownLabel),
                }
            }
        }
    };
}

metric_enum!(MlMetricKind {
    Classification => "classification",
    Regression => "regression",
    Ranking => "ranking",
    Clustering => "clustering",
    Forecasting => "forecasting",
    Generation => "generation",
    Retrieval => "retrieval",
    Calibration => "calibration",
    Fairness => "fairness",
    Performance => "performance",
    Resource => "resource",
    Other => "other",
});

metric_enum!(MlMetricDirection {
    HigherIsBetter => "higher-is-better",
    LowerIsBetter => "lower-is-better",
    TargetIsBest => "target-is-best",
    Unknown => "unknown",
});

metric_enum!(MlMetricAggregation {
    Mean => "mean",
    Median => "median",
    Min => "min",
    Max => "max",
    Sum => "sum",
    WeightedMean => "weighted-mean",
    Macro => "macro",
    Micro => "micro",
    Samples => "samples",
    None => "none",
});

metric_enum!(MlClassificationMetric {
    Accuracy => "accuracy",
    Precision => "precision",
    Recall => "recall",
    F1 => "f1",
    RocAuc => "roc-auc",
    PrAuc => "pr-auc",
    LogLoss => "log-loss",
    MatthewsCorrelationCoefficient => "matthews-correlation-coefficient",
    BalancedAccuracy => "balanced-accuracy",
});

impl MlClassificationMetric {
    pub const fn direction(self) -> MlMetricDirection {
        match self {
            Self::LogLoss => MlMetricDirection::LowerIsBetter,
            Self::Accuracy
            | Self::Precision
            | Self::Recall
            | Self::F1
            | Self::RocAuc
            | Self::PrAuc
            | Self::MatthewsCorrelationCoefficient
            | Self::BalancedAccuracy => MlMetricDirection::HigherIsBetter,
        }
    }
}

metric_enum!(MlRegressionMetric {
    Mae => "mae",
    Mse => "mse",
    Rmse => "rmse",
    R2 => "r2",
    Mape => "mape",
    Smape => "smape",
    MedianAbsoluteError => "median-absolute-error",
});

impl MlRegressionMetric {
    pub const fn direction(self) -> MlMetricDirection {
        match self {
            Self::R2 => MlMetricDirection::HigherIsBetter,
            Self::Mae
            | Self::Mse
            | Self::Rmse
            | Self::Mape
            | Self::Smape
            | Self::MedianAbsoluteError => MlMetricDirection::LowerIsBetter,
        }
    }
}

metric_enum!(MlRankingMetric {
    Ndcg => "ndcg",
    Map => "map",
    Mrr => "mrr",
    HitRate => "hit-rate",
    RecallAtK => "recall-at-k",
    PrecisionAtK => "precision-at-k",
});

impl MlRankingMetric {
    pub const fn direction(self) -> MlMetricDirection {
        MlMetricDirection::HigherIsBetter
    }
}

metric_enum!(MlClusteringMetric {
    Silhouette => "silhouette",
    AdjustedRandIndex => "adjusted-rand-index",
    NormalizedMutualInfo => "normalized-mutual-info",
    DaviesBouldin => "davies-bouldin",
});

impl MlClusteringMetric {
    pub const fn direction(self) -> MlMetricDirection {
        match self {
            Self::DaviesBouldin => MlMetricDirection::LowerIsBetter,
            Self::Silhouette | Self::AdjustedRandIndex | Self::NormalizedMutualInfo => {
                MlMetricDirection::HigherIsBetter
            },
        }
    }
}

metric_enum!(MlGenerationMetric {
    Bleu => "bleu",
    Rouge => "rouge",
    Meteor => "meteor",
    BertScore => "bert-score",
    ExactMatch => "exact-match",
    Perplexity => "perplexity",
});

impl MlGenerationMetric {
    pub const fn direction(self) -> MlMetricDirection {
        match self {
            Self::Perplexity => MlMetricDirection::LowerIsBetter,
            Self::Bleu | Self::Rouge | Self::Meteor | Self::BertScore | Self::ExactMatch => {
                MlMetricDirection::HigherIsBetter
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MlMetricError {
    Empty,
    NonFinite,
    UnknownLabel,
}

impl fmt::Display for MlMetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("ML metric metadata text cannot be empty"),
            Self::NonFinite => formatter.write_str("ML metric value must be finite"),
            Self::UnknownLabel => formatter.write_str("unknown ML metric metadata label"),
        }
    }
}

impl Error for MlMetricError {}

fn non_empty_text(value: impl AsRef<str>) -> Result<String, MlMetricError> {
    let trimmed = value.as_ref().trim();
    if trimmed.is_empty() {
        Err(MlMetricError::Empty)
    } else {
        Ok(trimmed.to_string())
    }
}

fn normalized_label(value: &str) -> Result<String, MlMetricError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(MlMetricError::Empty)
    } else {
        Ok(trimmed.to_ascii_lowercase().replace(['_', ' '], "-"))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MlClassificationMetric, MlMetricDirection, MlMetricError, MlMetricName, MlMetricValue,
        MlRankingMetric, MlRegressionMetric,
    };

    #[test]
    fn validates_metric_names_and_values() -> Result<(), MlMetricError> {
        let name = MlMetricName::new(" accuracy ")?;
        let value = MlMetricValue::new(0.93)?;

        assert_eq!(name.as_str(), "accuracy");
        assert_eq!(value.value(), 0.93);
        assert_eq!(MlMetricName::new("  "), Err(MlMetricError::Empty));
        assert_eq!(MlMetricValue::new(f64::NAN), Err(MlMetricError::NonFinite));
        Ok(())
    }

    #[test]
    fn displays_parses_and_labels_metric_directions() -> Result<(), MlMetricError> {
        assert_eq!(
            "roc auc".parse::<MlClassificationMetric>()?,
            MlClassificationMetric::RocAuc
        );
        assert_eq!(
            "precision at k".parse::<MlRankingMetric>()?,
            MlRankingMetric::PrecisionAtK
        );
        assert_eq!(
            "rmse".parse::<MlRegressionMetric>()?,
            MlRegressionMetric::Rmse
        );
        assert_eq!(
            MlClassificationMetric::Accuracy.direction(),
            MlMetricDirection::HigherIsBetter
        );
        assert_eq!(
            MlClassificationMetric::LogLoss.direction(),
            MlMetricDirection::LowerIsBetter
        );
        assert_eq!(
            MlRegressionMetric::R2.direction(),
            MlMetricDirection::HigherIsBetter
        );
        assert_eq!(
            MlRegressionMetric::Rmse.direction(),
            MlMetricDirection::LowerIsBetter
        );
        Ok(())
    }
}
