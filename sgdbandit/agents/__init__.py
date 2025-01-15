from .abstract_agent import AbstractAgent
from .sgd_smom import SGD_SMoM
from .ucb_agents import ClassicUCB, RobustUCBCatoni, RobustUCBMedian, RobustUCBTruncated
from .heavy_inf import HeavyInf
from .adaptive_inf import AdaptiveInf
from .ape import APE
from .ape_numba import APENumba
from .tsallis_med_inf import ClippedMedSmd
from .tsallis_inf import Tsallis_INF
__all__ = [
    "AbstractAgent",
    "ClassicUCB",
    "RobustUCBTruncated",
    "RobustUCBCatoni",
    "RobustUCBMedian",
    "SGD_SMoM",
    "HeavyInf",
    "AdaptiveInf",
    "ClippedMedSmd",
    # "APENumba", bagged version
    "Tsallis_INF"
]
