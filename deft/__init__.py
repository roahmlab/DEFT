"""
DEFT: Differentiable Branched Discrete Elastic Rods for Modeling Furcated DLOs in Real-Time

A framework for modeling Branched Deformable Linear Objects (BDLOs) using differentiable physics-based models.
"""

__version__ = "1.0.0"
__author__ = "Yizhou Chen, Xiaoyue Wu, Yeheng Zong, et al."

from .core import DEFT_sim, DEFT_func
from .utils import util

__all__ = ["DEFT_sim", "DEFT_func", "util"]