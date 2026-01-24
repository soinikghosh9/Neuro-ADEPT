#!/usr/bin/env python3
"""
Neuro-ADEPT: Anomaly Detection via Error-driven Predictive Temporal Processing
for EEG Seizure Detection

Optimized for RTX 5070 8GB GPU - Targeting Clinical Performance

Key Features:
1. Deep SVDD-inspired anomaly detection objective
2. VAE reconstruction-based anomaly scoring
3. Supervised contrastive representation learning
4. Neurological feature injection (band power, Hjorth, entropy)
5. Multi-Scale Temporal Convolution (captures different seizure durations)
6. Spectral Attention with gamma-band emphasis
7. Cross-Channel Spatial Attention (learns electrode correlations)
8. FiLM conditioning for context-aware processing

CORE PARADIGM: Seizure detection as ANOMALY DETECTION from learned normalcy.
The model learns what "normal" brain activity looks like and detects deviations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class PCNConfig:
    """Configuration for Neuro-ADEPT (Anomaly Detection via Error-driven Predictive Temporal).
    
    v4.3 Dual-Stream Architecture:
    - Simplified: Removed VAE and Contrastive heads (Loss Conflict Resolution)
    - Focused: Combined Deep SVDD + Predictive Coding for robustness
    - SVDD: Bias removal to prevent collapse
    """
    n_channels: int = 19
    n_samples: int = 1280
    n_classes: int = 2
    context_dim: int = 152

    # Architecture - Optimized for 8GB GPU with better capacity
    base_filters: int = 80       # Multi-scale spectral filterbank output
    hidden_dim: int = 128        # TCN hidden dimension (REDUCED from 160 to force overfitting)
    n_layers: int = 6            # TCN depth (receptive field ~4 seconds)
    kernel_size: int = 7         # Base kernel for EEG patterns
    dropout: float = 0.4         # INCREASED from 0.3 to combat overfitting
    drop_path: float = 0.2       # INCREASED from 0.1 for better regularization

    # Multi-scale kernels for temporal patterns
    multi_scale_kernels: tuple = (3, 7, 15)  # Short/medium/long patterns

    # Attention
    n_heads: int = 4

    # Anomaly Detection Parameters (Dual-Stream)
    use_anomaly_detection: bool = True     # Deep SVDD objective
    use_neuro_features: bool = True         # Neurological features (Hjorth, entropy)
    anomaly_weight: float = 0.6             # Primary anomaly mechanism
    svdd_nu: float = 0.2                    # Stronger margin enforcement

    # Training dynamics
    anomaly_warmup_epochs: int = 4


# =============================================================================
# NEUROLOGICAL FEATURE EXTRACTION MODULE
# =============================================================================

class NeuroFeatureExtractor(nn.Module):
    """
    Extracts clinically-relevant neurological features from EEG signals.

    v4.5 IMPROVED FEATURES for seizure discrimination:
    1. Band Power Ratio (high/low frequency): Increases during seizures
    2. Skewness: Asymmetry of amplitude distribution (seizures often skewed)
    3. Spectral Entropy: Decreases during organized seizure activity
    4. Hjorth Activity: Signal power (increases during seizures)
    5. Hjorth Mobility: Mean frequency (shifts during seizures)
    6. Hjorth Complexity: Bandwidth (changes with seizure onset)

    References:
    - Acharya et al. (2013) "Automated EEG Analysis of Epilepsy: A Review"
    - Shoeb (2009) MIT PhD Thesis on Seizure Detection
    - Tzallas et al. (2009) "Epileptic Seizure Detection in EEGs"
    """

    def __init__(self, n_channels: int, sample_rate: int = 256):
        super().__init__()
        self.n_channels = n_channels
        self.sample_rate = sample_rate

        # Learnable feature importance weights
        n_features = 6  # band_ratio, skewness, entropy, activity, mobility, complexity
        self.feature_weights = nn.Parameter(torch.ones(n_features))

        # Feature projection (combines with main features)
        self.proj = nn.Sequential(
            nn.Linear(n_channels * n_features, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract neurological features from raw EEG.

        Args:
            x: (B, C, T) raw EEG signal

        Returns:
            features: (B, 32) neurological feature vector
        """
        B, C, T = x.shape

        # 1. Spectral Derivative Log-Ratio (Band Power Ratio proxy)
        # High-frequency content increases during seizures
        dx = x[:, :, 1:] - x[:, :, :-1]  # First derivative
        d2x = dx[:, :, 1:] - dx[:, :, :-1]  # Second derivative

        high_freq_power = torch.var(d2x, dim=2) + 1e-8  # Approximates gamma
        low_freq_power = torch.var(x, dim=2) + 1e-8     # Approximates delta
        band_ratio = torch.log(high_freq_power / low_freq_power)  # Log ratio

        # 2. Skewness (REPLACES line_length - better seizure discriminator)
        # Seizures often produce skewed amplitude distributions
        # Formula: E[(X-μ)³] / σ³
        mean_x = x.mean(dim=2, keepdim=True)
        std_x = x.std(dim=2, keepdim=True) + 1e-8
        x_centered = (x - mean_x) / std_x
        skewness = torch.mean(x_centered ** 3, dim=2)  # Third moment
        # Clamp for numerical stability
        skewness = torch.clamp(skewness, -10, 10)

        # 3. Spectral Entropy approximation (using signal variance distribution)
        # Lower entropy during organized seizure activity
        window_size = T // 8
        if window_size > 0:
            x_windows = x.unfold(2, window_size, window_size)  # (B, C, n_windows, window_size)
            window_var = torch.var(x_windows, dim=3) + 1e-8  # (B, C, n_windows)
            window_prob = window_var / (window_var.sum(dim=2, keepdim=True) + 1e-8)
            entropy = -torch.sum(window_prob * torch.log(window_prob + 1e-8), dim=2)
        else:
            entropy = torch.zeros(B, C, device=x.device)

        # 4. Hjorth Activity (signal variance - increases during seizures)
        activity = torch.var(x, dim=2)
        activity = torch.log(activity + 1e-8)

        # 5. Hjorth Mobility (mean frequency indicator)
        mobility = torch.sqrt(torch.var(dx, dim=2) / (torch.var(x, dim=2) + 1e-8) + 1e-8)

        # 6. Hjorth Complexity (bandwidth measure)
        d2x_var = torch.var(d2x, dim=2)
        dx_var = torch.var(dx, dim=2) + 1e-8
        x_var = torch.var(x, dim=2) + 1e-8
        complexity = torch.sqrt(d2x_var / dx_var + 1e-8) / (torch.sqrt(dx_var / x_var + 1e-8) + 1e-8)

        # Stack features: (B, C, 6)
        features = torch.stack([
            band_ratio * self.feature_weights[0],
            skewness * self.feature_weights[1],      # CHANGED from line_length
            entropy * self.feature_weights[2],
            activity * self.feature_weights[3],
            mobility * self.feature_weights[4],
            complexity * self.feature_weights[5]
        ], dim=2)

        # Flatten and project: (B, C*6) -> (B, 32)
        features = features.reshape(B, -1)
        features = self.proj(features)

        return features


# =============================================================================
# ANOMALY DETECTION MODULES
# =============================================================================

class DeepSVDDHead(nn.Module):
    """
    Deep Support Vector Data Description (SVDD) head for anomaly detection.

    Core idea: Learn a hypersphere in latent space that contains normal EEG.
    Seizures (anomalies) will map outside this hypersphere.

    Reference: Ruff et al. (2018) "Deep One-Class Classification" ICML

    v4.5 ROBUST CENTER UPDATE:
    - Outlier rejection using Median Absolute Deviation (MAD)
    - Adaptive EMA momentum based on sample count
    - Center validation to prevent NaN/Inf corruption
    - Running radius estimation for adaptive margin
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        super().__init__()

        # Deeper mapping network φ(x) for better representation
        # [CRITICAL] bias=False to prevent hypersphere collapse to a trivial point
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim)  # Normalize embeddings for stable distances
        )

        # Learnable center c (initialized from NORMAL samples only)
        self.register_buffer('center', torch.zeros(output_dim))
        self.register_buffer('center_count', torch.tensor(0.0))
        self.register_buffer('running_radius', torch.tensor(1.0))  # v4.5: Track radius
        self.center_initialized = False
        self.output_dim = output_dim

        # Adaptive EMA momentum (increases with sample count for stability)
        self.base_momentum = 0.95
        self.min_momentum = 0.9
        self.max_momentum = 0.999

    def _compute_robust_center(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        v4.5: Compute robust center using outlier rejection.
        Uses Median Absolute Deviation (MAD) to filter outliers.
        """
        n_samples = embeddings.shape[0]
        
        if n_samples < 3:
            return embeddings.mean(dim=0)
        
        # Compute per-dimension median
        median = embeddings.median(dim=0).values
        
        # Compute MAD (Median Absolute Deviation) for each dimension
        abs_dev = torch.abs(embeddings - median)
        mad = abs_dev.median(dim=0).values + 1e-8
        
        # Compute z-scores using MAD (more robust than std)
        # MAD to std conversion factor for normal distribution: 1.4826
        z_scores = abs_dev / (1.4826 * mad)
        
        # Filter outliers: keep samples with max z-score < 3
        max_z_per_sample = z_scores.max(dim=1).values
        inlier_mask = max_z_per_sample < 3.0
        
        if inlier_mask.sum() >= 2:
            # Use inliers only
            return embeddings[inlier_mask].mean(dim=0)
        else:
            # Fallback to median if too few inliers
            return median

    def _get_adaptive_momentum(self) -> float:
        """
        v4.5: Compute adaptive momentum based on sample count.
        Early updates use lower momentum (faster adaptation).
        Later updates use higher momentum (stability).
        """
        count = self.center_count.item()
        
        if count < 100:
            return self.min_momentum
        elif count < 1000:
            # Linear interpolation
            t = (count - 100) / 900
            return self.min_momentum + t * (self.base_momentum - self.min_momentum)
        elif count < 10000:
            t = (count - 1000) / 9000
            return self.base_momentum + t * (self.max_momentum - self.base_momentum)
        else:
            return self.max_momentum

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) feature vector
            labels: (B,) optional labels for center initialization (0=normal, 1=seizure)

        Returns:
            embedding: (B, output_dim) SVDD embedding
            anomaly_score: (B,) distance from center (higher = more anomalous)
        """
        embedding = self.encoder(x)

        # v4.5 ROBUST CENTER UPDATE
        min_normal_samples = 4  # Need minimum samples for robust estimation
        
        if self.training and labels is not None:
            normal_mask = (labels == 0)
            n_normal = normal_mask.sum().item()
            
            if n_normal >= min_normal_samples:
                normal_embeddings = embedding[normal_mask].detach()
                
                # Use robust center computation with outlier rejection
                batch_center = self._compute_robust_center(normal_embeddings)
                
                # Validate center (prevent NaN/Inf corruption)
                if torch.isfinite(batch_center).all():
                    if not self.center_initialized:
                        # First initialization
                        self.center = batch_center.clone()
                        self.center_initialized = True
                        self.center_count += n_normal
                        
                        # Initialize running radius
                        dists = torch.sum((normal_embeddings - batch_center) ** 2, dim=1)
                        self.running_radius = dists.mean().sqrt() + 1e-6
                    else:
                        # Adaptive EMA update of center
                        momentum = self._get_adaptive_momentum()
                        self.center = momentum * self.center + (1 - momentum) * batch_center
                        self.center_count += n_normal
                        
                        # Update running radius (for adaptive margin)
                        dists = torch.sum((normal_embeddings - self.center) ** 2, dim=1)
                        batch_radius = dists.mean().sqrt()
                        self.running_radius = 0.99 * self.running_radius + 0.01 * batch_radius

        # Anomaly score = squared distance from center (normalized by dimension)
        diff = embedding - self.center
        anomaly_score = torch.sum(diff ** 2, dim=1) / self.output_dim

        return embedding, anomaly_score

    def compute_loss(self, anomaly_scores: torch.Tensor, labels: torch.Tensor,
                     nu: float = 0.1, warmup_done: bool = True) -> torch.Tensor:
        """
        Compute Deep SVDD loss with soft-boundary formulation.

        Args:
            anomaly_scores: (B,) squared distances from center
            labels: (B,) 0=normal, 1=seizure
            nu: Fraction of outliers (controls soft boundary)
            warmup_done: Whether warmup period is complete
        """
        if not warmup_done:
            # During warmup, only minimize distance for normal samples (no push-away)
            is_normal = (labels == 0).float()
            return (anomaly_scores * is_normal).mean() * 0.5

        is_normal = (labels == 0).float()
        is_seizure = (labels == 1).float()

        # For normal: minimize distance (pull toward center)
        loss_normal = anomaly_scores * is_normal

        # v4.5: Use running radius for more stable margin
        # Margin = radius + 2*std (approximate 95th percentile)
        if (labels == 0).any():
            normal_scores = anomaly_scores[labels == 0].detach()
            if normal_scores.numel() > 2:
                std_val = normal_scores.std()
                if torch.isnan(std_val) or std_val < 1e-6:
                    std_val = torch.tensor(0.1, device=normal_scores.device)
                # Use running radius as base, add 2*std for margin
                margin = (self.running_radius ** 2 / self.output_dim) + 2 * std_val
                margin = margin.clamp(min=0.1, max=10.0)  # Prevent extreme margins
            elif normal_scores.numel() > 0:
                margin = normal_scores.mean() + 0.2
            else:
                margin = self.running_radius ** 2 / self.output_dim
        else:
            margin = self.running_radius ** 2 / self.output_dim

        # Hinge loss: penalize if seizure score is BELOW margin
        loss_seizure = F.relu(margin - anomaly_scores) * is_seizure

        # Balanced combination
        n_normal = is_normal.sum().clamp(min=1)
        n_seizure = is_seizure.sum().clamp(min=1)

        loss = loss_normal.sum() / n_normal + nu * loss_seizure.sum() / n_seizure

        return loss


# [REMOVED] ReconstructionHead and ContrastiveLossHead classes (Simplification)


def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class SpectralFilterbank(nn.Module):
    """
    Enhanced Multi-scale Spectral Filterbank for EEG seizure detection.

    [v3.8] IMPROVEMENTS:
    - Deeper per-branch processing for better feature extraction
    - Residual connections within branches
    - Temporal context aggregation
    - Stronger gamma emphasis for seizure detection

    At 256Hz sampling rate:
    - Kernel 4 -> ~64Hz bandwidth (gamma) - CRITICAL for seizures
    - Kernel 16 -> ~16Hz bandwidth (beta/alpha)
    - Kernel 64 -> ~4Hz bandwidth (delta/theta) - background
    """
    def __init__(self, n_channels: int, n_filters: int, kernel_size: int = 32):
        super().__init__()

        # More filters for gamma (seizure-critical)
        gamma_filters = n_filters // 2  # 50% for gamma
        beta_filters = n_filters // 4   # 25% for beta
        delta_filters = n_filters - gamma_filters - beta_filters  # 25% for delta

        # High-frequency branch (gamma band - MOST IMPORTANT for seizures)
        # Use multiple kernel sizes within gamma range for better resolution
        self.conv_gamma = nn.Sequential(
            nn.Conv1d(n_channels, gamma_filters, 4, padding=2, bias=False),
            nn.BatchNorm1d(gamma_filters),
            nn.GELU(),
            nn.Conv1d(gamma_filters, gamma_filters, 3, padding=1, groups=gamma_filters, bias=False),
            nn.BatchNorm1d(gamma_filters),
            nn.GELU()
        )

        # Mid-frequency branch (beta/alpha)
        self.conv_beta = nn.Sequential(
            nn.Conv1d(n_channels, beta_filters, 16, padding=8, bias=False),
            nn.BatchNorm1d(beta_filters),
            nn.GELU(),
            nn.Conv1d(beta_filters, beta_filters, 5, padding=2, groups=beta_filters, bias=False),
            nn.BatchNorm1d(beta_filters),
            nn.GELU()
        )

        # Low-frequency branch (delta/theta)
        self.conv_delta = nn.Sequential(
            nn.Conv1d(n_channels, delta_filters, 64, padding=32, bias=False),
            nn.BatchNorm1d(delta_filters),
            nn.GELU(),
            nn.Conv1d(delta_filters, delta_filters, 7, padding=3, groups=delta_filters, bias=False),
            nn.BatchNorm1d(delta_filters),
            nn.GELU()
        )

        # Fusion with temporal context
        self.fusion = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, 1, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.GELU(),
            nn.Conv1d(n_filters, n_filters, 3, padding=1, groups=n_filters, bias=False),  # Temporal context
            nn.BatchNorm1d(n_filters),
            nn.GELU()
        )

        # Learnable band weighting (strongly emphasize gamma for seizures)
        self.band_weights = nn.Parameter(torch.tensor([1.5, 1.0, 0.7]))  # gamma >> beta > delta

        # Spectral-temporal attention - learns to weight both bands and time
        self.spectral_attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, n_filters // 2),
            nn.GELU(),
            nn.Linear(n_filters // 2, n_filters),
            nn.Sigmoid()
        )

        # Temporal attention for local context
        self.temporal_context = nn.Sequential(
            nn.Conv1d(n_filters, n_filters, 7, padding=3, groups=n_filters, bias=False),
            nn.BatchNorm1d(n_filters),
            nn.Sigmoid()
        )

        # Downsample
        self.pool = nn.AvgPool1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[2]

        # Multi-scale feature extraction with proper alignment
        f_gamma = self.conv_gamma(x)[:, :, :T] * self.band_weights[0]
        f_beta = self.conv_beta(x)[:, :, :T] * self.band_weights[1]
        f_delta = self.conv_delta(x)[:, :, :T] * self.band_weights[2]

        # Concatenate multi-scale features
        f = torch.cat([f_gamma, f_beta, f_delta], dim=1)
        f = self.fusion(f)

        # Apply spectral attention (channel-wise)
        spec_attn = self.spectral_attn(f).unsqueeze(2)  # (B, C, 1)
        f = f * spec_attn

        # Apply temporal attention (time-wise gating)
        temp_attn = self.temporal_context(f)  # (B, C, T)
        f = f * temp_attn

        return self.pool(f)


class CrossChannelAttention(nn.Module):
    """
    v4.5 IMPROVED: Learns spatial correlations between EEG channels
    while preserving temporal dynamics.
    
    Previous issue: Pooled all time steps before attention, losing temporal info.
    Fix: Use strided temporal segments for efficient attention that preserves dynamics.
    
    Important for seizure localization patterns which evolve over time.
    """
    def __init__(self, n_channels: int, n_heads: int = 4, n_segments: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.n_segments = n_segments  # Split time into segments for attention
        self.head_dim = max(1, n_channels // n_heads)
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(n_channels, n_channels * 3, bias=False)
        self.proj = nn.Linear(n_channels, n_channels)
        
        # Temporal mixing layer to combine segment-wise attention
        self.temporal_mix = nn.Conv1d(n_channels, n_channels, kernel_size=3, 
                                       padding=1, groups=n_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        B, C, T = x.shape
        
        # Split time into segments for efficient attention
        seg_len = max(1, T // self.n_segments)
        n_segs = T // seg_len
        
        if n_segs > 1:
            # Reshape to (B, C, n_segs, seg_len) and pool within segments
            x_segs = x[:, :, :n_segs * seg_len].reshape(B, C, n_segs, seg_len)
            x_seg_pooled = x_segs.mean(dim=3)  # (B, C, n_segs)
            
            # Compute attention for each segment
            seg_outputs = []
            for s in range(n_segs):
                x_s = x_seg_pooled[:, :, s]  # (B, C)
                
                # Multi-head attention on this segment
                qkv = self.qkv(x_s).reshape(B, 3, self.n_heads, -1)
                qkv = qkv.permute(1, 0, 2, 3)  # (3, B, heads, head_dim)
                q, k, v = qkv[0], qkv[1], qkv[2]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)

                out_s = (attn @ v).reshape(B, C)
                seg_outputs.append(self.proj(out_s))
            
            # Stack and expand back to time dimension
            seg_stack = torch.stack(seg_outputs, dim=2)  # (B, C, n_segs)
            
            # Interpolate back to full time dimension
            out = F.interpolate(seg_stack, size=T, mode='linear', align_corners=False)
            
            # Apply temporal mixing for smooth transitions
            out = self.temporal_mix(out)
        else:
            # Fallback for very short sequences
            x_pooled = x.mean(dim=2)  # (B, C)
            qkv = self.qkv(x_pooled).reshape(B, 3, self.n_heads, -1)
            qkv = qkv.permute(1, 0, 2, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).reshape(B, C)
            out = self.proj(out).unsqueeze(2).expand(-1, -1, T)

        return x + out


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation for context conditioning.
    More expressive than simple addition.
    """
    def __init__(self, context_dim: int, n_channels: int):
        super().__init__()
        self.gamma = nn.Linear(context_dim, n_channels)
        self.beta = nn.Linear(context_dim, n_channels)

        # Initialize to identity transformation
        nn.init.ones_(self.gamma.weight.data[:, 0])
        nn.init.zeros_(self.gamma.weight.data[:, 1:])
        nn.init.zeros_(self.gamma.bias.data)
        nn.init.zeros_(self.beta.weight.data)
        nn.init.zeros_(self.beta.bias.data)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), context: (B, context_dim)
        gamma = self.gamma(context).unsqueeze(2)  # (B, C, 1)
        beta = self.beta(context).unsqueeze(2)
        return gamma * x + beta


class CausalConv1d(nn.Module):
    """Causal Convolution (no future leakage)."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, max(1, in_channels // reduction)),
            nn.ReLU(),
            nn.Linear(max(1, in_channels // reduction), in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.fc(x).unsqueeze(2)
        return x * attn


class MultiScaleTCNBlock(nn.Module):
    """
    Multi-scale TCN block with FiLM conditioning and DropPath.
    Captures patterns at different temporal scales.
    """
    def __init__(self, n_channels: int, hidden_dim: int, kernel_sizes: tuple,
                 dilation: int, context_dim: int, dropout: float = 0.2,
                 drop_path_rate: float = 0.0):
        super().__init__()

        n_kernels = len(kernel_sizes)
        hidden_per_kernel = hidden_dim // n_kernels

        # [FIX] Multi-scale parallel convolutions with remainder handling
        # Ensure sum(out_channels) == hidden_dim exactly (128 // 3 = 42 -> 126 total, missing 2)
        self.convs = nn.ModuleList()
        current_ch = 0
        
        for i, k in enumerate(kernel_sizes):
            if i == n_kernels - 1:
                out_ch = hidden_dim - current_ch # Give remainder to last kernel
            else:
                out_ch = hidden_per_kernel
            
            self.convs.append(
                CausalConv1d(n_channels, out_ch, k, dilation=dilation)
            )
            current_ch += out_ch

        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # FiLM conditioning
        self.film = FiLMLayer(context_dim, hidden_dim)

        # Output projection
        self.conv_out = nn.Conv1d(hidden_dim, n_channels, 1)
        self.norm2 = nn.BatchNorm1d(n_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Channel attention
        self.se = SEBlock(n_channels)

        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = x

        # Multi-scale convolution
        conv_outs = [conv(x) for conv in self.convs]

        # Ensure all outputs have the same time dimension
        min_t = min(out.shape[2] for out in conv_outs)
        conv_outs = [out[:, :, :min_t] for out in conv_outs]

        out = torch.cat(conv_outs, dim=1)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        # FiLM conditioning
        out = self.film(out, context)

        # Output projection
        out = self.conv_out(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        # Channel attention
        out = self.se(out)

        # Residual with stochastic depth
        # Handle potential size mismatch
        if residual.shape[2] != out.shape[2]:
            residual = residual[:, :, :out.shape[2]]

        return residual + self.drop_path(out)


class TemporalAttention(nn.Module):
    """
    Enhanced Temporal Attention with multi-head support.
    Pools time steps based on learned relevance.
    """
    def __init__(self, in_channels: int, n_heads: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.n_heads = n_heads

        self.query = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_heads)
        )

        # Value projection for richer representations
        self.value_proj = nn.Linear(in_channels, in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        B, C, T = x.shape
        t = x.transpose(1, 2)  # (B, T, C)

        # Multi-head attention scores
        attn_scores = self.query(t)  # (B, T, n_heads)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, n_heads)

        # Average across heads for final attention
        attn_avg = attn_weights.mean(dim=2, keepdim=True)  # (B, T, 1)

        # Project values
        v = self.value_proj(t)  # (B, T, C)

        # Weighted sum
        context = torch.sum(v * attn_avg, dim=1)  # (B, C)

        return context, attn_avg


class PredictiveCodingCore(nn.Module):
    """
    Core predictive coding module with multi-scale temporal processing.
    """
    def __init__(self, config: PCNConfig):
        super().__init__()

        # Context mapper with normalization
        self.ctx_mapper = nn.Sequential(
            nn.Linear(config.context_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # Stochastic depth schedule (linear increase)
        dpr = [x.item() for x in torch.linspace(0, config.drop_path, config.n_layers)]

        # Multi-scale TCN layers
        self.layers = nn.ModuleList([
            MultiScaleTCNBlock(
                n_channels=config.base_filters,
                hidden_dim=config.hidden_dim,
                kernel_sizes=config.multi_scale_kernels,
                dilation=2 ** i,
                context_dim=config.hidden_dim,
                dropout=config.dropout,
                drop_path_rate=dpr[i]
            )
            for i in range(config.n_layers)
        ])

        # Cross-channel attention (applied once for efficiency)
        self.channel_attn = CrossChannelAttention(config.base_filters, n_heads=config.n_heads)

        # Predictor head for self-supervised objective
        self.predictor = nn.Sequential(
            nn.Conv1d(config.base_filters, config.base_filters, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.base_filters, config.base_filters, 1)
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx_emb = self.ctx_mapper(context)

        out = x
        for layer in self.layers:
            out = layer(out, ctx_emb)

        # Apply cross-channel attention
        out = self.channel_attn(out)

        # Prediction
        preds = self.predictor(out)

        # Compute prediction error (delta=4 to avoid identity mapping)
        delta = 4
        if out.shape[2] > delta:
            target = x[:, :, delta:out.shape[2]]
            prediction = preds[:, :, :out.shape[2]-delta]
            error = target - prediction
            error = F.pad(error, (delta, 0))
        else:
            error = torch.zeros_like(out)

        return out, error


class LightPCN(nn.Module):
    """
    Neuro-ADEPT v4.3: Dual-Stream Anomaly Detection (SVDD + Prediction)
    Optimized for RTX 5070 8GB GPU.

    Architecture:
    1. Neurological Feature Extraction (band power, entropy, Hjorth)
    2. Multi-scale Spectral Filterbank
    3. Predictive Coding Core with FiLM conditioning
    4. Multi-head Temporal Attention
    5. Deep SVDD Anomaly Detection Head
    6. Robust classifier with dual-stream fusion

    CORE INSIGHT: Dual-stream anomaly detection
    - Stream 1 (Temporal): Prediction errors capture onset dynamics (state bifurcation)
    - Stream 2 (Feature): SVDD captures deviation from normal hypersphere (state anomaly)
    """
    def __init__(self, config: Optional[PCNConfig] = None):
        super().__init__()
        if config is None:
            config = PCNConfig()
        self.config = config

        # 0. Neurological Feature Extractor (NEW in v4)
        if config.use_neuro_features:
            self.neuro_features = NeuroFeatureExtractor(config.n_channels)
            neuro_feat_dim = 32
        else:
            self.neuro_features = None
            neuro_feat_dim = 0

        # 1. Multi-scale Spectral Filterbank
        self.filterbank = SpectralFilterbank(
            config.n_channels,
            config.base_filters,
            kernel_size=32
        )

        # 2. Predictive Coding Core
        self.pcn = PredictiveCodingCore(config)

        # 3. Temporal Attention
        self.attn = TemporalAttention(
            config.base_filters,
            n_heads=config.n_heads,
            hidden_dim=config.hidden_dim
        )

        # Feature dimension after attention pooling
        feat_dim = config.base_filters + neuro_feat_dim

        # 4. Deep SVDD Head (NEW in v4) - Anomaly Detection
        if config.use_anomaly_detection:
            self.svdd_head = DeepSVDDHead(
                input_dim=feat_dim,
                hidden_dim=64,
                output_dim=32
            )
        else:
            self.svdd_head = None

        # 5. Reconstruction Head [REMOVED]
        self.recon_head = None

        # 6. Contrastive Head [REMOVED]
        self.contrastive_head = None

        # 7. Multi-signal Fusion Classifier (Dual-Stream)
        # Combines: spectral features + neuro features + anomaly scores
        # Additional features from anomaly heads
        anomaly_feat_dim = 0
        if config.use_anomaly_detection:
            anomaly_feat_dim += 1  # SVDD anomaly score only

        classifier_input_dim = feat_dim + anomaly_feat_dim + config.base_filters # + error_feat (per channel)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Final classification layer with special initialization
        self.final_fc = nn.Linear(config.hidden_dim // 2, config.n_classes)

        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))

        # Store dimensions for forward pass
        self.feat_dim = feat_dim
        self.neuro_feat_dim = neuro_feat_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # CRITICAL: Initialize final classification layer to predict uniform distribution
        # This prevents the model from immediately biasing toward one class
        nn.init.zeros_(self.final_fc.weight)
        nn.init.zeros_(self.final_fc.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                warmup_done: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass with anomaly-aware processing.

        Args:
            x: (B, n_channels, n_samples) raw EEG
            context: (B, context_dim) contextual features
            labels: (B,) optional labels for computing auxiliary losses during training
            warmup_done: (bool) whether warmup period is complete for anomaly losses

        Returns:
            logits: (B, n_classes) classification logits
            pred_loss: (B,) predictive coding loss per sample
            extras: Dict with attention, anomaly scores, etc.
        """
        # 0. Extract neurological features
        if self.neuro_features is not None:
            neuro_feats = self.neuro_features(x)  # (B, 32)
        else:
            neuro_feats = None

        # 1. Spectral feature extraction
        feats = self.filterbank(x)  # (B, base_filters, T/4)

        # 2. Predictive coding
        out, error = self.pcn(feats, context)

        # 3. Temporal attention pooling
        pooled, attn_weights = self.attn(out)  # (B, base_filters), (B, T, 1)

        # 4. Combine spectral + neurological features
        if neuro_feats is not None:
            combined_feats = torch.cat([pooled, neuro_feats], dim=1)
        else:
            combined_feats = pooled

        # 5. Anomaly Detection Heads
        extras = {
            'attn': attn_weights,
            'error_map': error,
            'latent': pooled,
            'neuro_feats': neuro_feats
        }

        anomaly_features = []
        anomaly_losses = {}

        # Deep SVDD anomaly score - pass labels for proper center initialization
        if self.svdd_head is not None:
            svdd_emb, anomaly_score = self.svdd_head(combined_feats, labels)
            anomaly_features.append(anomaly_score.unsqueeze(1))
            extras['svdd_embedding'] = svdd_emb
            extras['anomaly_score'] = anomaly_score

            if labels is not None:
                anomaly_losses['svdd'] = self.svdd_head.compute_loss(
                    anomaly_score, labels, nu=self.config.svdd_nu, warmup_done=warmup_done
                )

        # Removed: Reconstruction and Contrastive heads

        # [v5.0 FIX] Inject Prediction Error (Surprisal) into Classifier
        # The classifier needs to KNOW that the prediction failed (high error = seizure).
        # We compute mean absolute error per channel.
        # error: (B, C, T) -> error_feat: (B, C)
        # v5.7: Enhanced Surprisal (Mean + Max)
        # Seizures can be short. Mean dilutes the signal. Max captures the peak violation.
        # We combine both to catch sustained AND transient anomalies.
        err_abs = torch.abs(error)
        error_feat = (torch.mean(err_abs, dim=2) + torch.max(err_abs, dim=2)[0]) * 0.5
        
        # 6. Fuse anomaly features with main features for classification
        if anomaly_features:
            anomaly_concat = torch.cat(anomaly_features, dim=1)
            # Adaptive normalization: use sigmoid to bound scores to [0, 1]
            # Then scale to provide meaningful signal without dominating
            anomaly_normalized = torch.sigmoid(anomaly_concat - anomaly_concat.mean()) * 2 - 1
            # Combine: Features + Error + Anomaly Scores
            classifier_input = torch.cat([combined_feats, error_feat, anomaly_normalized], dim=1)
        else:
            classifier_input = torch.cat([combined_feats, error_feat], dim=1)

        # 7. Classification with temperature scaling
        features = self.classifier(classifier_input)
        logits = self.final_fc(features) / self.temperature

        # 8. Prediction loss (clamped for stability)
        error_clamped = torch.clamp(error, -5.0, 5.0)
        pred_loss = torch.mean(F.smooth_l1_loss(
            error_clamped,
            torch.zeros_like(error_clamped),
            reduction='none'
        ), dim=[1, 2])

        extras['anomaly_losses'] = anomaly_losses

        return logits, pred_loss, extras

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing for class imbalance."""
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                            label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        return focal


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Imbalanced Classification (ASL) - v4.2 TUNED.

    Based on: "Asymmetric Loss For Multi-Label Classification" (ICCV 2021)
    Modified for binary seizure detection with clinical priority.

    v4.2 TUNING:
    - gamma_pos=0.5: Slight focusing on hard seizures (misclassified ones)
    - gamma_neg=2.0: Reduced from 3.0 - less aggressive normal down-weighting
    - positive_weight=4.0: Increased seizure emphasis

    CLINICAL RATIONALE:
    - Missing a seizure is much worse than a false alarm
    - Therefore seizure loss is weighted more heavily (positive_weight)
    - Moderate gamma_neg ensures model still learns robust normal patterns
    """
    def __init__(self, gamma_pos: float = 0.5, gamma_neg: float = 2.0,
                 clip: float = 0.05, eps: float = 1e-8, positive_weight: float = 4.0):
        super().__init__()
        self.gamma_pos = gamma_pos  # Slight focusing on hard seizure samples
        self.gamma_neg = gamma_neg  # Moderate focusing for easy negatives
        self.clip = clip  # Small clip to reduce easy negative contribution
        self.eps = eps
        self.positive_weight = positive_weight  # Strong weight on seizure class

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, 2)
            targets: Labels (B,) with 0=normal, 1=seizure
        """
        probs = torch.softmax(inputs, dim=1)

        # Probability of seizure class
        p_seizure = probs[:, 1]  # (B,)

        # Create target mask
        is_seizure = (targets == 1).float()
        is_normal = 1 - is_seizure

        # For seizure samples: standard cross-entropy (gamma=0 means no focal reduction)
        # Every seizure sample contributes fully to learning
        # Loss = -y * (1-p)^gamma * log(p) with gamma=0 => -y * log(p)
        p_seizure_clamped = torch.clamp(p_seizure, self.eps, 1 - self.eps)
        loss_pos = -is_seizure * ((1 - p_seizure_clamped) ** self.gamma_pos) * torch.log(p_seizure_clamped)
        # Apply extra weight to seizure samples (clinical priority)
        loss_pos = loss_pos * self.positive_weight

        # For normal samples: focal loss with probability clipping
        # Easy negatives (high p_normal) contribute less
        p_normal = 1 - p_seizure
        # Clip: if model is very confident about normal, reduce its contribution
        p_normal_clipped = torch.clamp(p_normal - self.clip, 0.0, 1.0 - self.clip)
        p_normal_clipped = p_normal_clipped / (1.0 - self.clip)  # Renormalize
        p_normal_clamped = torch.clamp(p_normal_clipped, self.eps, 1 - self.eps)
        loss_neg = -is_normal * ((1 - p_normal_clamped) ** self.gamma_neg) * torch.log(p_normal_clamped)

        loss = loss_pos + loss_neg
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Classification + Prediction + Anomaly Detection Loss (Neuro-ADEPT v4.3).

    Loss Components (Simplified):
    1. Classification: Hybrid ASL + Focal Loss
    2. Prediction: Predictive coding loss (self-supervised)
    3. Deep SVDD: Hypersphere anomaly detection

    v4.3 IMPROVEMENTS:
    - Removed VAE and Contrastive losses (Loss Conflict Resolution)
    - Faster warmup ramp
    """
    def __init__(self, weight=None, lambda_pred=0.0, gamma=2.0, label_smoothing=0.05,
                 use_asymmetric: bool = True, use_hybrid: bool = False,
                 anomaly_weight: float = 0.6, gamma_neg: float = 1.5):
        super().__init__()
        self.use_asymmetric = use_asymmetric
        self.use_hybrid = use_hybrid

        # Base anomaly detection loss weights
        self.base_anomaly_weight = anomaly_weight

        # Current weights (can be dynamically adjusted)
        self.anomaly_weight = anomaly_weight
        self.recon_weight = 0.0 # Unused
        self.contrastive_weight = 0.0 # Unused

        if use_hybrid and weight is not None:
            # HYBRID: Use both AsymmetricLoss AND weighted focal
            pos_w = weight[1].item() if weight is not None else 1.0
            self.asl = AsymmetricLoss(gamma_pos=0.5, gamma_neg=gamma_neg, clip=0.02, positive_weight=pos_w)
            self.focal = FocalLoss(weight=weight, gamma=gamma, label_smoothing=label_smoothing)
            self.cls_loss_fn = self._hybrid_loss
        elif use_asymmetric:
            # Pure asymmetric loss
            # v5.8 FIX: Use the calculated class weight for seizures
            pos_w = weight[1].item() if weight is not None else 1.0
            print(f"  [CombinedLoss] ASL Positive Weight: {pos_w:.2f}")
            self.cls_loss_fn = AsymmetricLoss(gamma_pos=0.5, gamma_neg=gamma_neg, clip=0.02, positive_weight=pos_w)
            self.focal = self.cls_loss_fn
        else:
            # Standard focal loss with class weights
            self.cls_loss_fn = FocalLoss(weight=weight, gamma=gamma,
                                         label_smoothing=label_smoothing)
            self.focal = self.cls_loss_fn

        self.ce = self.cls_loss_fn  # For mixup
        self.lambda_pred = lambda_pred

    def set_epoch(self, epoch: int, total_epochs: int, warmup_epochs: int = 4):
        """
        Dynamically adjust loss weights based on training progress.

        v4.2 Strategy (FASTER warmup):
        - During warmup: 50%→100% of base weights (was 30%→100%)
        - After warmup: 100%→120% for maximum anomaly sensitivity
        """
        if epoch <= warmup_epochs:
            # Faster warmup: 50% -> 100% over warmup period
            progress = epoch / warmup_epochs
            scale = 0.5 + 0.5 * progress  # 50% -> 100% (was 30% -> 100%)
        else:
            # Post-warmup: Full weights, gradually increasing
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            scale = 1.0 + 0.2 * progress  # 100% -> 120% as training progresses

        self.anomaly_weight = self.base_anomaly_weight * scale
        # Unused weights set to 0
        self.recon_weight = 0.0
        self.contrastive_weight = 0.0

    def _hybrid_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combine ASL and weighted Focal for maximum effect."""
        return 0.6 * self.asl(logits, targets) + 0.4 * self.focal(logits, targets)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                pred_loss: torch.Tensor,
                anomaly_losses: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss with anomaly detection components.
        """
        cls_loss = self.cls_loss_fn(logits, targets)
        
        # [v5.0 FIX] Mask Prediction Loss (True Anomaly Detection)
        # Only minimize prediction error for NORMAL samples (targets == 0).
        # Ensures seizures remain "surprising" (high error) by not training on them.
        is_normal = (targets == 0).float()
        p_loss = (pred_loss * is_normal).sum() / (is_normal.sum().clamp(min=1))

        total = cls_loss + self.lambda_pred * p_loss

        loss_dict = {'cls': cls_loss.item(), 'pred': p_loss.item()}

        # Add anomaly detection losses with dynamic weights
        if anomaly_losses is not None:
            if 'svdd' in anomaly_losses:
                svdd_loss = anomaly_losses['svdd']
                total = total + self.anomaly_weight * svdd_loss
                loss_dict['svdd'] = svdd_loss.item()

            # Removed: Reconstruction and Contrastive losses

        return total, loss_dict


def get_model_info(model: LightPCN) -> str:
    n_params = model.count_parameters()
    config = model.config

    # Build feature list
    features = []
    if config.use_neuro_features:
        features.append("NeuroFeats-v4.5")  # Updated: skewness instead of line_length
    if config.use_anomaly_detection:
        features.append("DeepSVDD-BiasFree")

    feat_str = "+".join(features) if features else "None"

    return f'''
NEURO-ADEPT v4.5 (Improved Features + Temporal-Aware Attention)

Input:  {config.n_channels}ch @ {config.n_samples} samples | Context: {config.context_dim}D
Filterbank: Multi-scale Spectral ({config.base_filters} filters)
TCN: {config.n_layers} layers, kernels {config.multi_scale_kernels}
Attention: Segment-Based Cross-Channel + Multi-Head Temporal
Features: Skewness+Kurtosis (replaces line_length)
Anomaly Detection: {feat_str}
Anomaly Weights: SVDD={config.anomaly_weight}
Warmup Epochs: {config.anomaly_warmup_epochs} (dynamic loss scaling)
Regularization: Dropout={config.dropout}, DropPath={config.drop_path}
Parameters: {n_params:,} ({n_params/1e6:.2f}M)
'''
