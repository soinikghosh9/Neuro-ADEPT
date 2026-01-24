#!/usr/bin/env python3
"""
Publication-Ready Visualization Module for Neuro-PCN

All plots are designed for:
- IEEE/EMBC conference paper format
- High DPI (300+) output
- Clear legends outside plot area
- Non-overlapping titles and labels
- Consistent color scheme
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
from pathlib import Path

# Publication-ready style settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Consistent color palette
COLORS = {
    'pathology': '#D62728',    # Red - seizure/pathology
    'interictal': '#1F77B4',   # Blue - normal/interictal
    'highlight': '#FF7F0E',    # Orange - attention/focus
    'secondary': '#2CA02C',    # Green - secondary highlight
    'neutral': '#7F7F7F',      # Gray - neutral elements
}


def plot_binary_results(all_targets, all_preds, all_probs, output_dir, fold_num):
    """
    Generate publication-ready binary classification plots.

    Creates:
    1. Confusion Matrix with percentages
    2. ROC Curve with clinical operating points
    3. Precision-Recall Curve with F1 iso-lines
    """
    output_dir = Path(output_dir)

    # ========================================
    # 1. CONFUSION MATRIX (Enhanced)
    # ========================================
    cm = confusion_matrix(all_targets, all_preds)
    # Safe normalization - avoid division by zero if a class has no samples
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.where(row_sums > 0, cm.astype('float') / row_sums * 100, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))

    # Create heatmap with both counts and percentages
    sns.heatmap(cm, annot=False, cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Seizure'],
                yticklabels=['Normal', 'Seizure'],
                cbar_kws={'label': 'Count', 'shrink': 0.8})

    # Add text annotations with count and percentage
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_normalized[i, j]
            color = 'white' if pct > 50 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{count:,}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=10, color=color, fontweight='bold')

    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix - Fold {fold_num}', fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_dir / f'cm_fold{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # 2. ROC CURVE (Enhanced with clinical points)
    # ========================================
    y_score = all_probs[:, 1]
    fpr, tpr, thresholds = roc_curve(all_targets, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Main ROC curve
    ax.plot(fpr, tpr, color=COLORS['pathology'], lw=2,
            label=f'ROC (AUC = {roc_auc:.3f})')

    # Reference diagonal
    ax.plot([0, 1], [0, 1], color=COLORS['neutral'], lw=1.5, linestyle='--',
            label='Random (AUC = 0.500)')

    # Mark clinical operating points
    # Point 1: 90% Specificity (FPR = 0.1)
    idx_90spec = np.argmin(np.abs(fpr - 0.1))
    ax.scatter([fpr[idx_90spec]], [tpr[idx_90spec]], s=80, c=COLORS['highlight'],
               marker='o', zorder=5, edgecolors='black', linewidths=1)
    ax.annotate(f'90% Spec\nSens={tpr[idx_90spec]:.2f}',
                xy=(fpr[idx_90spec], tpr[idx_90spec]),
                xytext=(fpr[idx_90spec] + 0.15, tpr[idx_90spec] - 0.1),
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    # Point 2: 80% Sensitivity (TPR = 0.8)
    idx_80sens = np.argmin(np.abs(tpr - 0.8))
    if tpr[idx_80sens] > 0.5:  # Only show if achievable
        ax.scatter([fpr[idx_80sens]], [tpr[idx_80sens]], s=80, c=COLORS['secondary'],
                   marker='s', zorder=5, edgecolors='black', linewidths=1)
        ax.annotate(f'80% Sens\nSpec={1-fpr[idx_80sens]:.2f}',
                    xy=(fpr[idx_80sens], tpr[idx_80sens]),
                    xytext=(fpr[idx_80sens] + 0.15, tpr[idx_80sens] + 0.05),
                    fontsize=8, ha='left',
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title(f'ROC Curve - Fold {fold_num}', fontweight='bold', pad=10)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / f'roc_fold{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========================================
    # 3. PRECISION-RECALL CURVE (Enhanced)
    # ========================================
    precision, recall, _ = precision_recall_curve(all_targets, y_score)
    pr_auc = auc(recall, precision)

    # Calculate baseline (proportion of positives)
    baseline = np.sum(all_targets) / len(all_targets)

    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Main PR curve
    ax.plot(recall, precision, color=COLORS['pathology'], lw=2,
            label=f'PR Curve (AUC = {pr_auc:.3f})')

    # Baseline (random classifier)
    ax.axhline(y=baseline, color=COLORS['neutral'], lw=1.5, linestyle='--',
               label=f'Baseline (Prevalence = {baseline:.2f})')

    # Add F1 iso-lines
    f1_scores = [0.2, 0.4, 0.6, 0.8]
    for f1 in f1_scores:
        x = np.linspace(0.01, 1, 100)
        # F1 iso-line formula: y = f1*x / (2x - f1), undefined when 2x = f1
        denominator = 2 * x - f1
        # Use np.divide with 'where' to avoid computing division for zero/small denominators
        y = np.full_like(x, np.nan)
        valid_mask = np.abs(denominator) > 1e-6
        np.divide(f1 * x, denominator, out=y, where=valid_mask)
        y[y < 0] = np.nan
        y[y > 1] = np.nan
        ax.plot(x, y, color='gray', alpha=0.3, linestyle=':', lw=1)
        # Label F1 lines
        valid_idx = np.where(~np.isnan(y))[0]
        if len(valid_idx) > 0:
            mid_idx = valid_idx[len(valid_idx)//2]
            ax.annotate(f'F1={f1}', xy=(x[mid_idx], y[mid_idx]),
                       fontsize=7, color='gray', alpha=0.7)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - Fold {fold_num}', fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / f'pr_fold{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_neuro_interpretability(attn_weights, error_map, targets, output_dir, fold_num, probs=None):
    """
    Publication-ready Neural Interpretability Visualization.

    Shows seizure risk assessment based on predictive coding principles:
    - Risk Score: Clinical decision support metric
    - Free Energy: Prediction error from PCN
    - Temporal Attention: Model focus areas
    """
    output_dir = Path(output_dir)

    seizure_idxs = np.where(targets == 1)[0]
    normal_idxs = np.where(targets == 0)[0]

    if len(seizure_idxs) == 0:
        print("  [PLOT] Warning: No seizure samples for interpretability plot")
        return

    # Select most interesting seizure sample
    best_idx = seizure_idxs[0]
    max_var = -1
    for idx in seizure_idxs[:min(50, len(seizure_idxs))]:
        var = np.var(attn_weights[idx])
        if var > max_var:
            max_var = var
            best_idx = idx

    attn = attn_weights[best_idx].flatten()
    error = np.mean(error_map[best_idx]**2, axis=0)

    # Compute Risk Score
    error_iqr = np.percentile(error, 75) - np.percentile(error, 25)
    if error_iqr > 0:
        risk_zscore = (error - np.median(error)) / error_iqr
        risk_score = 100 / (1 + np.exp(-risk_zscore))
    else:
        risk_score = np.ones_like(error) * 50

    T = len(attn)
    time = np.linspace(0, 5, T)

    # Create figure with proper spacing
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1.2, 1.2, 0.6], hspace=0.35)

    # ========================================
    # Panel 1: SEIZURE RISK SCORE
    # ========================================
    ax1 = fig.add_subplot(gs[0])

    ax1.fill_between(time, 0, risk_score, alpha=0.3, color=COLORS['pathology'])
    ax1.plot(time, risk_score, color=COLORS['pathology'], lw=2, label='Seizure Risk')
    ax1.axhline(y=70, color=COLORS['highlight'], linestyle='--', lw=1.5,
                label='Clinical Threshold (70%)')

    # Mark high-risk regions
    high_risk = risk_score > 70
    if np.any(high_risk):
        ax1.fill_between(time, 70, risk_score, where=high_risk,
                         alpha=0.5, color=COLORS['pathology'])

    ax1.set_ylabel('Risk Score (%)', fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.set_xlim([0, 5])
    ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    ax1.set_title('A) Seizure Risk Score (Clinical Decision Support)',
                  fontweight='bold', loc='left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticklabels([])

    # ========================================
    # Panel 2: FREE ENERGY (Prediction Error)
    # ========================================
    ax2 = fig.add_subplot(gs[1])

    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-6)
    ax2.fill_between(time, 0, error_norm, alpha=0.3, color='#9467BD')
    ax2.plot(time, error_norm, color='#9467BD', lw=2, label='Prediction Error')

    ax2.set_ylabel('Free Energy\n(Normalized)', fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.set_xlim([0, 5])
    ax2.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    ax2.set_title('B) Predictive Coding Surprisal Signal',
                  fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticklabels([])

    # ========================================
    # Panel 3: TEMPORAL ATTENTION
    # ========================================
    ax3 = fig.add_subplot(gs[2])

    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
    ax3.bar(time, attn_norm, width=5/T, color=COLORS['highlight'], alpha=0.7,
            label='Attention Weight')

    ax3.set_ylabel('Attention', fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.set_xlim([0, 5])
    ax3.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    ax3.set_title('C) Temporal Attention (Model Focus)',
                  fontweight='bold', loc='left')
    ax3.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Neural Interpretability Analysis - Fold {fold_num}\n'
                 f'(Seizure Sample, Ground Truth: Pathology)',
                 fontweight='bold', y=1.02)

    plt.savefig(output_dir / f'neuro_explain_fold{fold_num}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Also create comparison plot
    if len(normal_idxs) > 0:
        _plot_comparison(attn_weights, error_map, best_idx, normal_idxs[0],
                        output_dir, fold_num)


def _plot_comparison(attn_weights, error_map, seizure_idx, normal_idx, output_dir, fold_num):
    """Side-by-side comparison of seizure vs normal sample."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    for col, (idx, label, color) in enumerate([
        (seizure_idx, 'Seizure', COLORS['pathology']),
        (normal_idx, 'Normal', COLORS['interictal'])
    ]):
        attn = attn_weights[idx].flatten()
        error = np.mean(error_map[idx]**2, axis=0)
        T = len(attn)
        time = np.linspace(0, 5, T)

        # Risk score
        error_iqr = np.percentile(error, 75) - np.percentile(error, 25)
        if error_iqr > 0:
            risk_zscore = (error - np.median(error)) / error_iqr
            risk_score = 100 / (1 + np.exp(-risk_zscore))
        else:
            risk_score = np.ones_like(error) * 50

        # Top: Risk Score
        ax = axes[0, col]
        ax.fill_between(time, 0, risk_score, alpha=0.3, color=color)
        ax.plot(time, risk_score, color=color, lw=2)
        ax.axhline(y=70, color=COLORS['highlight'], linestyle='--', lw=1, alpha=0.7)
        ax.set_ylabel('Risk (%)' if col == 0 else '')
        ax.set_ylim([0, 100])
        ax.set_xlim([0, 5])
        ax.set_title(f'{label} Sample\nMean Risk: {np.mean(risk_score):.0f}%',
                    fontweight='bold', color=color)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xticklabels([])

        # Bottom: Attention
        ax = axes[1, col]
        attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-6)
        ax.bar(time, attn_norm, width=5/T, color=color, alpha=0.7)
        ax.set_ylabel('Attention' if col == 0 else '')
        ax.set_xlabel('Time (s)')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 5])
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Seizure vs Normal Comparison - Fold {fold_num}',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'neuro_comparison_fold{fold_num}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_space(latent_features, targets, output_dir, fold_num):
    """
    Publication-ready latent space visualization using t-SNE.
    """
    output_dir = Path(output_dir)

    print("  [PLOT] Running t-SNE...")
    from sklearn.manifold import TSNE

    # Subsample for speed if needed
    n_samples = min(2000, len(targets))
    if len(targets) > n_samples:
        np.random.seed(42)
        idx = np.random.choice(len(targets), n_samples, replace=False)
        latent_sub = latent_features[idx]
        targets_sub = targets[idx]
    else:
        latent_sub = latent_features
        targets_sub = targets

    # Run t-SNE (use max_iter for sklearn >= 1.5, fallback to n_iter for older)
    try:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    embedding = tsne.fit_transform(latent_sub)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot each class
    for label, name, color, marker in [
        (0, 'Normal (Interictal)', COLORS['interictal'], 'o'),
        (1, 'Seizure (Pathology)', COLORS['pathology'], '^')
    ]:
        mask = targets_sub == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=color, label=name, alpha=0.6, s=20, marker=marker, edgecolors='none')

    ax.set_xlabel('t-SNE Dimension 1', fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontweight='bold')
    ax.set_title(f'Latent Space Visualization - Fold {fold_num}',
                fontweight='bold', pad=10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), framealpha=0.9, markerscale=1.5)

    # Remove axis ticks (t-SNE coordinates are arbitrary)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_dir / f'clustering_fold{fold_num}.png', dpi=300, bbox_inches='tight')
    plt.close()


def _compute_gradcam_for_class(model, data_sample, context_sample, target_class, device):
    """Compute Grad-CAM weights for a specific class."""
    import torch

    model.zero_grad()
    saved_tensors = {'activations': None, 'gradients': None}

    def save_activation_hook(_module, _input, output):
        activation = output.detach().clone()
        activation.requires_grad_(True)
        saved_tensors['activations'] = activation
        activation.register_hook(lambda grad: saved_tensors.update({'gradients': grad.clone()}))
        return activation

    forward_handle = model.filterbank.register_forward_hook(save_activation_hook)

    with torch.enable_grad():
        data_input = data_sample.detach().clone().requires_grad_(True)
        logits, _, _ = model(data_input, context_sample)
        class_logit = logits[0, target_class]
        class_logit.backward()

    forward_handle.remove()

    gradients = saved_tensors['gradients']
    activations = saved_tensors['activations']

    if gradients is None:
        return None

    weights = torch.mean(gradients * activations, dim=2)
    weights = torch.relu(weights)
    weights = weights.squeeze().detach().cpu().numpy()

    if weights.max() > 0:
        weights = weights / weights.max()

    return weights


def plot_gradcam_frequency_bands(model, val_loader, device, output_dir, fold_num):
    """
    Publication-ready Grad-CAM frequency band attribution.

    Shows which EEG frequency bands are most important for each class:
    - Seizure: Expected high gamma band attribution
    - Normal: Expected high alpha band attribution
    """
    import torch
    output_dir = Path(output_dir)

    model.eval()

    # Find confident samples from each class
    best_path_sample, best_path_prob = None, 0.0
    best_inter_sample, best_inter_prob = None, 0.0

    print("  [PLOT] Computing Grad-CAM frequency band attribution (both classes)...")

    with torch.no_grad():
        for data, context, target in val_loader:
            data, context, target = data.to(device), context.to(device), target.to(device)
            logits, _, _ = model(data, context)
            probs = torch.softmax(logits, dim=1)

            # Pathology samples
            path_mask = (target == 1)
            if path_mask.sum() > 0:
                path_probs = probs[path_mask, 1]
                max_idx = path_probs.argmax()
                if path_probs[max_idx].item() > best_path_prob:
                    best_path_prob = path_probs[max_idx].item()
                    idx = torch.where(path_mask)[0][max_idx]
                    best_path_sample = (data[idx:idx+1].clone(), context[idx:idx+1].clone())

            # Interictal samples
            inter_mask = (target == 0)
            if inter_mask.sum() > 0:
                inter_probs = probs[inter_mask, 0]
                max_idx = inter_probs.argmax()
                if inter_probs[max_idx].item() > best_inter_prob:
                    best_inter_prob = inter_probs[max_idx].item()
                    idx = torch.where(inter_mask)[0][max_idx]
                    best_inter_sample = (data[idx:idx+1].clone(), context[idx:idx+1].clone())

            if best_path_prob > 0.8 and best_inter_prob > 0.8:
                break

    # Compute Grad-CAM for both classes
    path_weights = None
    inter_weights = None

    if best_path_sample is not None:
        path_weights = _compute_gradcam_for_class(
            model, best_path_sample[0], best_path_sample[1], 1, device)

    if best_inter_sample is not None:
        inter_weights = _compute_gradcam_for_class(
            model, best_inter_sample[0], best_inter_sample[1], 0, device)

    if path_weights is None and inter_weights is None:
        print("  [PLOT] Warning: Could not compute Grad-CAM")
        return None

    # Frequency band definitions
    band_names = ['Delta\n(0.5-4Hz)', 'Theta\n(4-8Hz)', 'Alpha\n(8-13Hz)',
                  'Beta\n(13-30Hz)', 'Gamma\n(30-70Hz)']

    def compute_band_attrs(weights):
        if weights is None:
            return [0.0] * 5
        n_ch = len(weights)
        n_bands = 5
        band_sz = n_ch // n_bands
        return [np.mean(weights[i*band_sz:(i+1)*band_sz if i < 4 else n_ch])
                for i in range(n_bands)]

    path_bands = compute_band_attrs(path_weights)
    inter_bands = compute_band_attrs(inter_weights)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    x_pos = np.arange(len(band_names))
    width = 0.35

    # Panel 1: Seizure attribution
    ax = axes[0]
    bars = ax.bar(x_pos, path_bands, color=COLORS['pathology'], alpha=0.7,
                  edgecolor='black', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names, fontsize=8)
    ax.set_ylabel('Attribution Score', fontweight='bold')
    ax.set_title(f'A) Seizure Detection\n(Conf: {best_path_prob:.0%})',
                fontweight='bold', color=COLORS['pathology'])
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, path_bands):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Panel 2: Normal attribution
    ax = axes[1]
    bars = ax.bar(x_pos, inter_bands, color=COLORS['interictal'], alpha=0.7,
                  edgecolor='black', linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names, fontsize=8)
    ax.set_title(f'B) Normal Detection\n(Conf: {best_inter_prob:.0%})',
                fontweight='bold', color=COLORS['interictal'])
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, inter_bands):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Panel 3: Comparison
    ax = axes[2]
    ax.bar(x_pos - width/2, path_bands, width, label='Seizure',
           color=COLORS['pathology'], alpha=0.7, edgecolor='black')
    ax.bar(x_pos + width/2, inter_bands, width, label='Normal',
           color=COLORS['interictal'], alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_names, fontsize=8)
    ax.set_title('C) Comparison', fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle(f'Grad-CAM Frequency Band Attribution - Fold {fold_num}',
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / f'gradcam_freq_bands_fold{fold_num}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [PLOT] Grad-CAM: Pathology Gamma={path_bands[4]:.2f}, "
          f"Interictal Alpha={inter_bands[2]:.2f}")

    return {'pathology': path_bands, 'interictal': inter_bands}
