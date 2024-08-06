"""A library for quantum machine learning following the scikit-learnstandard."""

from . import encoding_circuit, kernel, observables, optimizers, qnn, util
from .util import Executor

__version__ = "0.7.5"

__all__ = [
    "Executor",
    "observables",
    "encoding_circuit",
    "kernel",
    "optimizers",
    "qnn",
    "util",
]
