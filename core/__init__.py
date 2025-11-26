"""
Kaggle Competitions Utilities Package

Core utilities for Kaggle competition notebooks supporting
local (conda), Google Colab, and Kaggle environments.
"""

__version__ = "0.1.0"

from .setup_env import KaggleEnvironment, setup

__all__ = ["KaggleEnvironment", "setup"]