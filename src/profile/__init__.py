"""通过路线成对选择学习 FAVOUR 四维用户画像。"""

from .exceptions import (
    ProfileError,
    ProfileNumericalError,
    ProfileStateError,
    ProfileValidationError,
)
from .inference import (
    BradleyTerryLogitLikelihood,
    FavourLaplacePosteriorEstimator,
    FavourPosteriorObjective,
    FavourPosteriorPredictor,
)
from .learner import PairwisePreferenceWeightLearner, WeightLearningConfig
from .models import (
    GaussianPreferencePrior,
    GroupPreferencePrior,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    PreferenceLearningResult,
    PreferencePosterior,
    RouteAttributes,
)
from .normalization import (
    NormalizationScale,
    NormalizedCostFeatureExtractor,
    RouteAttributeNormalizer,
)
from .priors import FixedGaussianPriorProvider, MassPreferencePriorEstimator
from .service import (
    FavourPreferenceLearningService,
    InMemoryPosteriorRepository,
    PosteriorRepository,
)

__all__ = [
    "BradleyTerryLogitLikelihood",
    "FavourLaplacePosteriorEstimator",
    "FavourPosteriorObjective",
    "FavourPosteriorPredictor",
    "FavourPreferenceLearningService",
    "FixedGaussianPriorProvider",
    "GaussianPreferencePrior",
    "GroupPreferencePrior",
    "InMemoryPosteriorRepository",
    "MassPreferencePriorEstimator",
    "NormalizationScale",
    "NormalizedCostFeatureExtractor",
    "PREFERENCE_DIMENSIONS",
    "PairwisePreference",
    "PairwisePreferenceWeightLearner",
    "PosteriorRepository",
    "PreferenceDimension",
    "PreferenceLearningResult",
    "PreferencePosterior",
    "ProfileError",
    "ProfileNumericalError",
    "ProfileStateError",
    "ProfileValidationError",
    "RouteAttributeNormalizer",
    "RouteAttributes",
    "WeightLearningConfig",
]
