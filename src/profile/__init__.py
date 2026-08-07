"""FAVOUR论文流程的四维缩减实现。"""

from .exceptions import ProfileNumericalError, ProfileValidationError
from .inference import (
    BradleyTerryLogitLikelihood,
    FavourPosteriorObjective,
    FavourPosteriorPredictor,
    MassPreferencePriorEstimator,
)
from .learner import (
    PairwisePreferenceWeightLearner,
    preference_prior_from_weights,
    preset_preference_prior,
    preset_preference_weights,
    standard_mass_preference_prior,
)
from .models import (
    GaussianPreferenceModel,
    PREFERENCE_DIMENSIONS,
    PairwisePreference,
    PreferenceDimension,
    PreferenceLearningResult,
    PreferencePreset,
    RouteAttributes,
)
from .normalization import NormalizedCostFeatureExtractor

__all__ = [
    "BradleyTerryLogitLikelihood",
    "FavourPosteriorObjective",
    "FavourPosteriorPredictor",
    "GaussianPreferenceModel",
    "MassPreferencePriorEstimator",
    "NormalizedCostFeatureExtractor",
    "PREFERENCE_DIMENSIONS",
    "PairwisePreference",
    "PairwisePreferenceWeightLearner",
    "PreferenceDimension",
    "PreferenceLearningResult",
    "PreferencePreset",
    "ProfileNumericalError",
    "ProfileValidationError",
    "RouteAttributes",
    "preference_prior_from_weights",
    "preset_preference_prior",
    "preset_preference_weights",
    "standard_mass_preference_prior",
]
