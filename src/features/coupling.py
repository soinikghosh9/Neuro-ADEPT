#!/usr/bin/env python3
"""
Cross-Frequency Coupling Feature Extraction

Implements efficient epoch-level features for seizure detection:
- Phase-Amplitude Coupling (PAC) via Modulation Index
- Amplitude-Amplitude Coupling (AAC)
- Band Power Features
- Statistical Features

All computations are vectorized for speed.
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class CFCConfig:
    """Configuration for CFC feature extraction."""
    fs: int = 256
    theta_band: Tuple[float, float] = (4.0, 8.0)
    gamma_band: Tuple[float, float] = (30.0, 70.0)
    n_phase_bins: int = 18  # For PAC computation
    

# ==============================================================================
# Core Signal Processing (Vectorized)
# ==============================================================================

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, 
                    fs: int, order: int = 4) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    
    if low >= high:
        return np.zeros_like(data)
    
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Handle edge cases
    if data.ndim == 1:
        if len(data) < 3 * order:
            return np.zeros_like(data)
        return signal.filtfilt(b, a, data)
    else:
        # Multi-channel: filter each channel
        result = np.zeros_like(data)
        for i in range(data.shape[0]):
            if len(data[i]) >= 3 * order:
                result[i] = signal.filtfilt(b, a, data[i])
        return result


def hilbert_envelope(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous amplitude and phase via Hilbert transform.
    
    Returns:
        amplitude: Instantaneous amplitude envelope
        phase: Instantaneous phase (radians, -π to π)
    """
    if data.ndim == 1:
        analytic = signal.hilbert(data)
        amplitude = np.abs(analytic)
        phase = np.angle(analytic)
    else:
        # Multi-channel
        amplitude = np.zeros_like(data)
        phase = np.zeros_like(data)
        for i in range(data.shape[0]):
            analytic = signal.hilbert(data[i])
            amplitude[i] = np.abs(analytic)
            phase[i] = np.angle(analytic)
    
    return amplitude, phase


# ==============================================================================
# Phase-Amplitude Coupling (PAC)
# ==============================================================================

def compute_modulation_index(phase: np.ndarray, amplitude: np.ndarray, 
                              n_bins: int = 18) -> float:
    """
    Compute Modulation Index (MI) for Phase-Amplitude Coupling.
    
    MI measures how strongly the amplitude of a high-frequency signal
    is modulated by the phase of a low-frequency signal.
    
    Based on Tort et al. (2010) - Measuring phase-amplitude coupling.
    
    Args:
        phase: Low-frequency phase signal (radians)
        amplitude: High-frequency amplitude envelope
        n_bins: Number of phase bins
        
    Returns:
        MI: Modulation Index (0 = no coupling, higher = stronger coupling)
    """
    # Bin edges from -π to π
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    
    # Compute mean amplitude in each phase bin
    mean_amp = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.sum(mask) > 0:
            mean_amp[i] = np.mean(amplitude[mask])
        else:
            mean_amp[i] = 0
    
    # Normalize to distribution
    total = np.sum(mean_amp)
    if total == 0:
        return 0.0
    
    p = mean_amp / total
    
    # Kullback-Leibler divergence from uniform distribution
    uniform = 1.0 / n_bins
    
    # Avoid log(0)
    p_safe = np.where(p > 0, p, 1e-10)
    
    kl_divergence = np.sum(p_safe * np.log(p_safe / uniform))
    
    # Normalize by log(N) to get MI in [0, 1]
    mi = kl_divergence / np.log(n_bins)
    
    return float(max(0, mi))


def compute_pac_features(theta_phase: np.ndarray, gamma_amp: np.ndarray,
                         n_bins: int = 18) -> Dict[str, float]:
    """
    Compute comprehensive PAC features.
    
    Returns:
        Dictionary with MI, mean vector length, preferred phase
    """
    # Modulation Index
    mi = compute_modulation_index(theta_phase, gamma_amp, n_bins)
    
    # Mean Vector Length (alternative PAC measure)
    # Compute complex vector: amplitude * exp(i * phase)
    z = gamma_amp * np.exp(1j * theta_phase)
    mvl = np.abs(np.mean(z)) / (np.mean(gamma_amp) + 1e-10)
    
    # Preferred phase (where gamma is strongest)
    preferred_phase = np.angle(np.mean(z))
    
    return {
        'modulation_index': mi,
        'mean_vector_length': float(mvl),
        'preferred_phase': float(preferred_phase)
    }


# ==============================================================================
# Band Power Features
# ==============================================================================

def compute_band_powers(data: np.ndarray, fs: int = 256) -> Dict[str, float]:
    """
    Compute relative band powers using FFT.
    
    Much faster than filtering + Hilbert for power estimation.
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70)
    }
    
    # FFT
    n = len(data)
    fft = np.abs(np.fft.rfft(data))
    freqs = np.fft.rfftfreq(n, 1/fs)
    power = fft ** 2
    total_power = np.sum(power) + 1e-10
    
    result = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_power = np.sum(power[mask])
        
        # Relative power
        result[f'{band_name}_power'] = float(band_power / total_power)
        
        # Log power (for regression)
        result[f'{band_name}_log'] = float(np.log10(band_power + 1e-10))
    
    # Cross-frequency ratios (important biomarkers)
    theta_power = np.sum(power[(freqs >= 4) & (freqs < 8)]) + 1e-10
    gamma_power = np.sum(power[(freqs >= 30) & (freqs < 70)]) + 1e-10
    alpha_power = np.sum(power[(freqs >= 8) & (freqs < 13)]) + 1e-10
    beta_power = np.sum(power[(freqs >= 13) & (freqs < 30)]) + 1e-10
    
    result['theta_gamma_ratio'] = float(np.log10(gamma_power / theta_power))
    result['alpha_beta_ratio'] = float(np.log10(beta_power / alpha_power))
    
    return result


# ==============================================================================
# Statistical Features  
# ==============================================================================

def compute_statistical_features(data: np.ndarray) -> Dict[str, float]:
    """Compute standard statistical features."""
    result = {}
    
    result['mean'] = float(np.mean(data))
    result['std'] = float(np.std(data))
    result['max_abs'] = float(np.max(np.abs(data)))
    result['min'] = float(np.min(data))
    result['max'] = float(np.max(data))
    
    # Higher-order statistics
    centered = data - np.mean(data)
    std = np.std(data) + 1e-10
    
    result['skewness'] = float(np.mean(centered**3) / std**3)
    result['kurtosis'] = float(np.mean(centered**4) / std**4 - 3)
    
    # Line length (activity indicator)
    result['line_length'] = float(np.mean(np.abs(np.diff(data))))
    
    # Zero crossings (frequency indicator)
    result['zero_crossings'] = float(np.sum(np.diff(np.sign(centered)) != 0))
    
    return result


# ==============================================================================
# Main Feature Extraction
# ==============================================================================

def extract_epoch_features(epoch: np.ndarray, fs: int = 256, 
                           config: Optional[CFCConfig] = None) -> np.ndarray:
    """
    Extract comprehensive features from a single EEG epoch.
    
    Features include:
    - Band powers (5 bands × 2 = 10 features per channel)
    - PAC metrics (3 features per channel)
    - AAC (1 feature per channel)
    - Statistical (9 features per channel)
    
    Total: ~23 features per channel
    
    Args:
        epoch: Shape (n_channels, n_samples) or (n_samples,)
        fs: Sampling frequency
        config: CFC configuration
        
    Returns:
        Feature vector (flat array)
    """
    if config is None:
        config = CFCConfig(fs=fs)
    
    # Handle 1D input
    if epoch.ndim == 1:
        epoch = epoch.reshape(1, -1)
    
    n_channels, n_samples = epoch.shape
    all_features = []
    
    for ch in range(n_channels):
        signal_ch = epoch[ch]
        
        # 1. Band Powers (fast, FFT-based)
        powers = compute_band_powers(signal_ch, fs)
        
        # 2. Filter for PAC
        try:
            theta_filt = bandpass_filter(signal_ch, config.theta_band[0], 
                                         config.theta_band[1], fs)
            gamma_filt = bandpass_filter(signal_ch, config.gamma_band[0], 
                                         config.gamma_band[1], fs)
            
            # Hilbert for instantaneous phase/amplitude
            theta_amp, theta_phase = hilbert_envelope(theta_filt)
            gamma_amp, gamma_phase = hilbert_envelope(gamma_filt)
            
            # 3. PAC Features
            pac = compute_pac_features(theta_phase, gamma_amp, config.n_phase_bins)
            
            # 4. AAC (amplitude-amplitude coupling)
            aac = float(np.corrcoef(theta_amp, gamma_amp)[0, 1])
            if np.isnan(aac):
                aac = 0.0
                
        except Exception:
            # Fallback if filtering fails
            pac = {'modulation_index': 0, 'mean_vector_length': 0, 'preferred_phase': 0}
            aac = 0.0
        
        # 5. Statistical Features
        stats = compute_statistical_features(signal_ch)
        
        # Combine all features for this channel
        ch_features = [
            # Band powers (10)
            powers['delta_power'], powers['theta_power'], powers['alpha_power'],
            powers['beta_power'], powers['gamma_power'],
            powers['delta_log'], powers['theta_log'], powers['alpha_log'],
            powers['beta_log'], powers['gamma_log'],
            # Ratios (2)
            powers['theta_gamma_ratio'], powers['alpha_beta_ratio'],
            # PAC (3)
            pac['modulation_index'], pac['mean_vector_length'], pac['preferred_phase'],
            # AAC (1)
            aac,
            # Stats (9)
            stats['mean'], stats['std'], stats['max_abs'],
            stats['skewness'], stats['kurtosis'],
            stats['line_length'], stats['zero_crossings'],
            stats['min'], stats['max']
        ]
        
        all_features.extend(ch_features)
    
    return np.array(all_features, dtype=np.float32)


def extract_features_batch(X: np.ndarray, fs: int = 256, 
                           config: Optional[CFCConfig] = None,
                           verbose: bool = True) -> np.ndarray:
    """
    Extract features from batch of epochs.
    
    Args:
        X: Shape (n_epochs, n_channels, n_samples)
        fs: Sampling frequency
        config: CFC configuration
        verbose: Print progress
        
    Returns:
        Feature matrix (n_epochs, n_features)
    """
    if config is None:
        config = CFCConfig(fs=fs)
    
    n_epochs = len(X)
    features_list = []
    
    for i in range(n_epochs):
        f = extract_epoch_features(X[i], fs, config)
        features_list.append(f)
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"    Extracted features: {i+1}/{n_epochs}")
    
    return np.array(features_list, dtype=np.float32)


# ==============================================================================
# Predictive Coding Model
# ==============================================================================

class PredictiveCFCModel:
    """
    Epoch-level Predictive Coding for Cross-Frequency Coupling.
    
    Learns the normal relationship between theta and gamma features,
    then detects anomalies as high prediction error (surprisal).
    """
    
    def __init__(self, regularization: float = 1e-4):
        self.regularization = regularization
        self.W = None  # Prediction weights
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        self._is_fitted = False
    
    def fit(self, theta_features: np.ndarray, gamma_features: np.ndarray):
        """
        Learn theta→gamma mapping from interictal data.
        
        Args:
            theta_features: Theta-related features (n_samples, n_theta_features)
            gamma_features: Gamma-related features (n_samples, n_gamma_features)
        """
        # Ridge regression: W = (X^T X + λI)^-1 X^T Y
        n_features = theta_features.shape[1]
        reg = self.regularization * np.eye(n_features)
        
        self.W = np.linalg.solve(
            theta_features.T @ theta_features + reg,
            theta_features.T @ gamma_features
        )
        
        # Compute baseline error statistics
        predicted = theta_features @ self.W
        errors = np.mean((gamma_features - predicted) ** 2, axis=1)
        
        self.baseline_mean = np.mean(errors)
        self.baseline_std = np.std(errors) + 1e-10
        
        self._is_fitted = True
    
    def compute_surprisal(self, theta_features: np.ndarray, 
                          gamma_features: np.ndarray) -> np.ndarray:
        """
        Compute surprisal (normalized prediction error).
        
        Returns:
            Surprisal scores (positive = more anomalous)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        predicted = theta_features @ self.W
        errors = np.mean((gamma_features - predicted) ** 2, axis=1)
        
        # Z-score normalization
        surprisal = (errors - self.baseline_mean) / self.baseline_std
        
        return surprisal


def split_features_theta_gamma(features: np.ndarray, n_channels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split feature matrix into theta-related and gamma-related features.
    
    Feature structure per channel (25 features):
    - 0-4: band powers (delta, theta, alpha, beta, gamma)
    - 5-9: log powers
    - 10-11: ratios
    - 12-14: PAC (MI, MVL, preferred_phase)
    - 15: AAC
    - 16-24: stats
    
    Theta features: theta power, theta log, ratios, PAC, AAC
    Gamma features: gamma power, gamma log, stats
    """
    features_per_channel = features.shape[1] // n_channels
    
    theta_indices = []
    gamma_indices = []
    
    for ch in range(n_channels):
        base = ch * features_per_channel
        
        # Theta-related: theta power (1), theta log (6), ratios (10-11), PAC (12-14), AAC (15)
        theta_indices.extend([base + 1, base + 6, base + 10, base + 11,
                              base + 12, base + 13, base + 14, base + 15])
        
        # Gamma-related: gamma power (4), gamma log (9), stats (16-24)
        gamma_indices.extend([base + 4, base + 9] + list(range(base + 16, base + 25)))
    
    return features[:, theta_indices], features[:, gamma_indices]
