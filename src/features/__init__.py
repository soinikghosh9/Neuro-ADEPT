"""
Feature extraction module for HSPR.

Provides epoch-level cross-frequency coupling features.
"""

from .coupling import (
    extract_epoch_features,
    extract_features_batch,
    compute_modulation_index,
    compute_pac_features,
    compute_band_powers,
    compute_statistical_features,
    PredictiveCFCModel,
    split_features_theta_gamma,
    CFCConfig
)

__all__ = [
    'extract_epoch_features',
    'extract_features_batch',
    'compute_modulation_index',
    'compute_pac_features',
    'compute_band_powers',
    'compute_statistical_features',
    'PredictiveCFCModel',
    'split_features_theta_gamma',
    'CFCConfig'
]
