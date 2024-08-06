"""QNN module for classification and regression."""

from .loss import (ConstantLoss, ParameterRegularizationLoss, SquaredLoss,
                   VarianceLoss)
from .qnnc import QNNClassifier
from .qnnr import QNNRegressor
from .training import ShotsFromRSTD, get_lr_decay, get_variance_fac

__all__ = [
    "SquaredLoss",
    "VarianceLoss",
    "ConstantLoss",
    "ParameterRegularizationLoss",
    "QNNClassifier",
    "QNNRegressor",
    "get_variance_fac",
    "get_lr_decay",
    "ShotsFromRSTD",
]
