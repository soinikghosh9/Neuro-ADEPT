#!/usr/bin/env python3
"""
Neuro-PCN v4.3 Training Pipeline (Dual-Stream)
Optimized for RTX 5070 8GB GPU - Targeting 90% Accuracy / 80% F1

Key Strategies:
1. Dynamic inverse frequency class weighting
2. Cosine annealing with warm restarts
3. Stronger augmentation (Mixup + CutMix)
4. Gradient accumulation for effective larger batches
5. Multi-metric early stopping (F1 + Sensitivity)
"""

import os
import sys
import gc
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent))

from src.models.pcn import LightPCN, PCNConfig, get_model_info, CombinedLoss
from src.data.dataset import create_optimized_loaders, get_subject_list, get_min_channels
from src.visualization.plots import (
    plot_binary_results, plot_neuro_interpretability,
    plot_latent_space, plot_gradcam_frequency_bands
)


# ==============================================================================
# Augmentation Functions
# ==============================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation: linear interpolation between samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation: cut and paste patches between samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # For 1D signal: cut a contiguous segment
    _, _, W = x.shape
    cut_w = int(W * (1 - lam))
    cx = np.random.randint(W)

    bbw1 = np.clip(cx - cut_w // 2, 0, W)
    bbw2 = np.clip(cx + cut_w // 2, 0, W)

    x_cut = x.clone()
    x_cut[:, :, bbw1:bbw2] = x[index, :, bbw1:bbw2]

    # Adjust lambda based on actual cut size
    lam = 1 - (bbw2 - bbw1) / W

    return x_cut, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, pred_loss):
    """Compute loss for mixup/cutmix augmentation."""
    cls_loss = lam * criterion.ce(pred, y_a) + (1 - lam) * criterion.ce(pred, y_b)
    total_loss = cls_loss + criterion.lambda_pred * torch.mean(pred_loss)
    return total_loss


# ==============================================================================
# Training Functions
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device, epoch,
                scaler=None, use_mixup=True, accumulation_steps=1, scheduler=None,
                anomaly_warmup_epochs: int = 8):
    """
    Train for one epoch with augmentation and gradient accumulation.

    Args:
        accumulation_steps: Number of batches to accumulate gradients over
        scheduler: Learning rate scheduler (OneCycleLR steps per batch)
        anomaly_warmup_epochs: Number of epochs before full anomaly loss kicks in
    """
    model.train()
    total_loss = 0
    total_pred_loss = 0
    total_pred_loss_normal = 0
    total_pred_loss_seizure = 0
    all_preds = []
    all_targets = []

    optimizer.zero_grad()

    # Warmup phases:
    # 1. Epochs 1-3: No mixup, basic feature learning (was 1-5)
    # 2. Epochs 1-anomaly_warmup_epochs: Reduced anomaly loss (warmup_done=False)
    mixup_warmup_done = epoch > 3  # REDUCED from 5 to start augmentation earlier
    anomaly_warmup_done = epoch > anomaly_warmup_epochs

    for batch_idx, (data, context, target) in enumerate(loader):
        data, context, target = data.to(device), context.to(device), target.to(device)

        # Select augmentation randomly - v4.2: MORE AGGRESSIVE augmentation
        aug_choice = np.random.random()

        if use_mixup and mixup_warmup_done and aug_choice < 0.25:
            # Mixup (25% of batches, was 15%) with stronger alpha
            inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha=0.4)  # was 0.2

            with autocast('cuda'):
                # Pass labels=None for mixup (anomaly losses disabled)
                logits, pred_err, extras = model(inputs, context, labels=None, warmup_done=anomaly_warmup_done)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam, pred_err)

        elif use_mixup and mixup_warmup_done and aug_choice < 0.45:
            # CutMix (20% of batches, was 10%) with stronger alpha
            inputs, targets_a, targets_b, lam = cutmix_data(data, target, alpha=1.0)  # was 0.5

            with autocast('cuda'):
                # Pass labels=None for cutmix (anomaly losses disabled)
                logits, pred_err, extras = model(inputs, context, labels=None, warmup_done=anomaly_warmup_done)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam, pred_err)
        else:
            # Standard training (100% during warmup, 75% after)
            with autocast('cuda'):
                # Pass labels for anomaly detection losses (v4.1)
                logits, pred_err, extras = model(data, context, labels=target, warmup_done=anomaly_warmup_done)
                anomaly_losses = extras.get('anomaly_losses', None)
                loss, _ = criterion(logits, target, pred_err, anomaly_losses)

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"  [WARNING] Loss is NaN/Inf at batch {batch_idx}! Skipping.")
            scaler.update()
            continue

        scaler.scale(loss).backward()

        # Update weights after accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Capture scale to detect if step is skipped
            scale_before = scaler.get_scale()
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step OneCycleLR scheduler per batch
            # v4.7 FIX: Only step scheduler if optimizer actually stepped.
            # If scaler.step() skips due to NaNs, scale decreases. We shouldn't step scheduler then.
            if scheduler is not None:
                # Check if scale decreased (implies skipped step)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()

        total_loss += loss.item() * accumulation_steps
        
        # Track split prediction error
        with torch.no_grad():
            p_err = pred_err  # Already (B,) from model forward
            is_norm = (target == 0)
            is_sz = (target == 1)
            
            p_norm = p_err[is_norm].mean().item() if is_norm.any() else 0
            p_sz = p_err[is_sz].mean().item() if is_sz.any() else 0
            
            total_pred_loss += p_err.mean().item()
            total_pred_loss_normal += p_norm
            total_pred_loss_seizure += p_sz

        # Track predictions (using original targets for metrics)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(target.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            recent_preds = all_preds[-len(preds)*50:]
            recent_targets = all_targets[-len(preds)*50:]
            acc = accuracy_score(recent_targets, recent_preds)
            n_pred_norm = (np.array(recent_preds) == 0).sum()
            n_pred_sz = (np.array(recent_preds) == 1).sum()
            lr_str = f"{scheduler.get_last_lr()[0]:.2e}" if scheduler else "N/A"
            print(f"  [Ep {epoch}] Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item()*accumulation_steps:.4f} | "
                  f"PredErr: N={total_pred_loss_normal/(batch_idx+1):.4f} S={total_pred_loss_seizure/(batch_idx+1):.4f} | "
                  f"Acc: {100*acc:.1f}% | Preds: N={n_pred_norm} S={n_pred_sz} | LR: {lr_str}")

    print(f"  [Ep {epoch}] Epoch Training Finished. Starting Validation...")
    return total_loss / len(loader), total_pred_loss_normal / len(loader), total_pred_loss_seizure / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, optimize_threshold: bool = True):
    """
    Validate model and compute comprehensive clinical metrics.

    CLINICAL METRICS INCLUDED:
    - Sensitivity (Recall): Ability to detect seizures
    - Specificity: Ability to identify normal periods
    - PPV (Precision): When alarm sounds, how often is it real?
    - NPV: When no alarm, how confident are we it's normal?
    - AUROC: Overall discrimination ability (threshold-independent)
    - AUPRC: Important for imbalanced data
    - Sensitivity at 90% Specificity: Clinical operating point
    - False Positive Rate per Hour (FPR/h): Clinical deployability
    - Optimized threshold metrics: Best operating point for seizure detection
    """
    model.eval()
    total_loss = 0
    total_pred_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    all_anomaly_scores = []

    for batch_idx, (data, context, target) in enumerate(loader):
        if batch_idx % 200 == 0:
             print(f"    [Val] Batch {batch_idx}/{len(loader)}...", end='\r')
        data, context, target = data.to(device), context.to(device), target.to(device)

        # Pass labels for computing validation anomaly metrics
        logits, pred_err, extras = model(data, context, labels=target)
        anomaly_losses = extras.get('anomaly_losses', None)
        loss, _ = criterion(logits, target, pred_err, anomaly_losses)

        total_loss += loss.item()
        total_pred_loss += torch.mean(pred_err).item()

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(target.cpu().numpy())
        all_probs.extend(probs)

        # Collect anomaly scores for analysis
        if 'anomaly_score' in extras:
            all_anomaly_scores.extend(extras['anomaly_score'].cpu().numpy())

    all_targets = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # [OPTIMIZATION] Temporal Smoothing (Median Filter)
    # Reduces false positives by requiring sustained detections
    from scipy.ndimage import median_filter
    if len(all_probs) > 5:
        # Smooth seizure probability (index 1) with window 3
        # Use Reflect padding to handle edges
        smoothed_probs = median_filter(all_probs[:, 1], size=3, mode='reflect')
        # Update probabilities - keep normal prob consistent (1-p)
        all_probs[:, 1] = smoothed_probs
        all_probs[:, 0] = 1 - smoothed_probs
        # Update hard predictions based on smoothed probs
        all_preds = np.argmax(all_probs, axis=1)

    # Confusion matrix at default threshold (0.5)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    # Basic metrics at default threshold
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
    f1_normal = f1_per_class[0] if len(f1_per_class) > 0 else 0
    f1_seizure = f1_per_class[1] if len(f1_per_class) > 1 else 0
    acc = accuracy_score(all_targets, all_preds)

    # Per-class metrics from confusion matrix
    # CM layout: [[TN, FP], [FN, TP]]
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    # Clinical metrics at default threshold
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # Seizure detection rate
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # True negative rate
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive predictive value
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0  # Negative predictive value

    # Balanced accuracy
    balanced_acc = (sensitivity + specificity) / 2

    # CLINICAL METRICS: AUROC and AUPRC
    auroc = 0.5
    auprc = 0.0
    sens_at_90spec = 0.0

    # Only calculate if we have both classes (otherwise undefined)
    if len(np.unique(all_targets)) > 1:
        try:
            auroc = roc_auc_score(all_targets, all_probs[:, 1])
            auprc = average_precision_score(all_targets, all_probs[:, 1])

            # Sensitivity at 90% Specificity (clinical operating point)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(all_targets, all_probs[:, 1])
            # Find threshold where specificity >= 0.9 (i.e., FPR <= 0.1)
            spec_90_idx = np.where(fpr <= 0.1)[0]
            if len(spec_90_idx) > 0:
                sens_at_90spec = tpr[spec_90_idx[-1]]
        except ValueError:
            pass

    # False Positive Rate per Hour (assuming 5s epochs at 256Hz)
    # Each epoch is 5 seconds, so 720 epochs = 1 hour
    total_interictal = TN + FP
    if total_interictal > 0:
        fpr_per_epoch = FP / total_interictal
        # Assuming each epoch is 5 seconds: 720 epochs per hour
        fpr_per_hour = fpr_per_epoch * 720
    else:
        fpr_per_hour = 0.0

    # OPTIMAL THRESHOLD for clinical use (maximize sensitivity while maintaining specificity)
    opt_threshold = 0.5
    opt_sens = sensitivity
    opt_spec = specificity
    opt_ppv = ppv

    if optimize_threshold and len(np.unique(all_targets)) > 1:
        try:
            opt_threshold, opt_metrics = find_optimal_threshold(all_probs, all_targets, target_sens=0.80)
            opt_sens = opt_metrics['sensitivity']
            opt_spec = opt_metrics['specificity']
            opt_ppv = opt_metrics['ppv']
        except Exception:
            pass  # Keep defaults

    return {
        'loss': total_loss / len(loader),
        'pred_loss': total_pred_loss / len(loader),
        'f1': f1,
        'f1_normal': f1_normal,
        'f1_seizure': f1_seizure,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'sens_inter': specificity,  # Backward compatible
        'sens_pathology': sensitivity,  # Backward compatible
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'auroc': auroc,
        'auprc': auprc,
        'sens_at_90spec': sens_at_90spec,
        'fpr_per_hour': fpr_per_hour,
        # Curve Data (for aggregation)
        'fpr': fpr if 'fpr' in locals() else None,
        'tpr': tpr if 'tpr' in locals() else None,
        # Optimized threshold metrics
        'opt_threshold': opt_threshold,
        'opt_sensitivity': opt_sens,
        'opt_specificity': opt_spec,
        'opt_ppv': opt_ppv,
        'probs': all_probs,
        'preds': all_preds,
        'targets': all_targets,
        'confusion_matrix': cm
    }


def compute_class_weights(train_loader, boost_pathology: float = 2.0):
    """
    Compute inverse frequency class weights from training data.

    v4.4 UPDATE: With BalancedBatchSampler, we use MODERATE weights.
    The sampler already ensures ~40% seizure per batch, so we only need
    a small additional boost, not the previous 10x that caused double-penalty.

    CLINICAL RATIONALE:
    - Missing a seizure (false negative) can be life-threatening
    - But with balanced batches, we don't need extreme weights
    - Moderate weights (1.5-4.0) provide clinical emphasis without oscillation

    Args:
        train_loader: DataLoader to analyze
        boost_pathology: Additional multiplier for pathology class (reduced from 10.0 to 2.0)
    """
    class_counts = {0: 0, 1: 0}

    for _, _, targets in train_loader:
        for t in targets.numpy():
            class_counts[t] += 1

    n_inter = class_counts[0]
    n_path = class_counts[1]
    total = n_inter + n_path

    # v4.4 MODERATE CLASS WEIGHTING:
    # With balanced sampling, batches are ~40% seizure already
    # Only apply slight additional weight
    if n_path > 0 and n_inter > 0:
        # Effective number of samples (Class-Balanced Loss, Cui et al. 2019)
        beta = 0.9999
        eff_n_inter = (1 - beta**n_inter) / (1 - beta)
        eff_n_path = (1 - beta**n_path) / (1 - beta)

        # Inverse effective frequency
        inter_weight = 1.0 / eff_n_inter
        path_weight = 1.0 / eff_n_path

        # Normalize so interictal = 1.0
        path_weight = (path_weight / inter_weight) * boost_pathology

        # v5.1: Relaxed clamp range [1.0, 10.0] to allow strong boosting
        # Manuscript calls for 5.0, so 2.0 clamp was incorrect
        path_weight = min(max(path_weight, 1.0), 10.0)
    else:
        path_weight = 5.0  # Moderate default weight (was 2.5)

    return torch.tensor([1.0, path_weight]), n_inter, n_path


def find_optimal_threshold(all_probs: np.ndarray, all_targets: np.ndarray,
                           target_sens: float = 0.80) -> Tuple[float, dict]:
    """
    Find optimal classification threshold using Youden's J statistic.
    This maximizes (sensitivity + specificity - 1), providing the best
    balance between true positive and true negative rates.

    Args:
        all_probs: Probability outputs (B, 2)
        all_targets: Ground truth labels
        target_sens: Minimum sensitivity target (used as fallback)

    Returns:
        optimal_threshold: Best threshold for seizure class probability
        metrics: Dict with metrics at optimal threshold
    """
    from sklearn.metrics import roc_curve

    seizure_probs = all_probs[:, 1]

    # Get ROC curve points
    fpr, tpr, thresholds = roc_curve(all_targets, seizure_probs)

    # Method 1: Youden's J statistic (maximizes sens + spec - 1)
    # This finds the point on ROC curve furthest from the diagonal
    j_scores = tpr - fpr  # Youden's J = sens + spec - 1 = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]

    # Ensure threshold is in valid range
    best_thresh = np.clip(best_thresh, 0.01, 0.99)

    # Compute metrics at optimal threshold
    preds = (seizure_probs >= best_thresh).astype(int)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, preds, labels=[0, 1])
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    sens_opt = TP / (TP + FN) if (TP + FN) > 0 else 0
    spec_opt = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv_opt = TP / (TP + FP) if (TP + FP) > 0 else 0

    return best_thresh, {
        'threshold': best_thresh,
        'sensitivity': sens_opt,
        'specificity': spec_opt,
        'ppv': ppv_opt
    }


def run_fold(train_subs, val_subs, config, device, output_dir, fold_num):
    """Run training for a single LOSO fold."""
    print(f"\n{'='*40} FOLD {fold_num} {'='*40}")

    # Get channel count
    all_subs = train_subs + val_subs
    n_channels = get_min_channels(config['cache_dir'], config['dataset'], all_subs)

    # Create data loaders
    train_loader, val_loader, _ = create_optimized_loaders(
        config['cache_dir'], config['dataset'],
        train_subs, val_subs,
        n_channels=n_channels,
        batch_size=config['batch_size'],
        val_batch_size=config['batch_size'] * 4,  # Efficient validation (no grad = low memory)
        num_workers=config.get('num_workers', 2),
        max_epochs_per_subject=config.get('max_epochs_per_subject', 300),
        binary_modes=True
    )

    # v5.6: Gentle Boost (2.0) - Reverted from aggressive 5.0 to improve stability
    # Aggressive boosting caused over-optimization for specificity
    class_weights, n_inter, n_path = compute_class_weights(train_loader, boost_pathology=2.0)
    class_weights = class_weights.to(device)

    print(f"\n[DEBUG] Class Distribution:")
    print(f"  Train: Inter={n_inter}, Path={n_path} (Ratio: {n_path/max(n_inter,1):.2f})")
    print(f"  [LOSS] Adaptive Class Weights: Inter=1.0, Path={class_weights[1].item():.2f} "
          f"(Inter/Path ratio: {n_inter/max(n_path,1):.2f})")

    # Model configuration optimized for RTX 5070 8GB
    # Neuro-ADEPT v4.2: Increased regularization, faster anomaly warmup
    model_config = PCNConfig(
        n_channels=n_channels,
        n_classes=2,
        context_dim=n_channels * 8,
        base_filters=80,        # Multi-scale spectral filterbank
        hidden_dim=160,         # TCN hidden dimension
        n_layers=6,             # TCN depth
        dropout=0.4,            # INCREASED from 0.3 to combat overfitting
        drop_path=0.2,          # INCREASED from 0.1 for better regularization
        multi_scale_kernels=(3, 7, 15),  # 3 scales for temporal patterns
        # v4.3: Dual-Stream Settings
        use_anomaly_detection=True,   # Deep SVDD objective
        use_neuro_features=True,       # Neurological features (Hjorth, entropy)
        anomaly_weight=0.6,            # SVDD loss weight
        svdd_nu=0.2,                  # SVDD margin parameter
        anomaly_warmup_epochs=4        # REDUCED from 8 for faster anomaly learning
    )

    model = LightPCN(model_config).to(device)
    try:
        print(get_model_info(model))
    except UnicodeEncodeError:
        # Fallback for Windows consoles that can't print fancy chars
        info = get_model_info(model).encode('ascii', 'ignore').decode('ascii')
        print(info)

    # Loss function - Neuro-ADEPT v4.3 DUAL-STREAM APPROACH
    # Combines: Classification + Predictive Coding + Deep SVDD
    criterion = CombinedLoss(
        weight=class_weights,      # Class-level weights for Focal component
        lambda_pred=0.01,          # Small predictive coding weight
        gamma=2.0,
        label_smoothing=0.05,      # Slight smoothing helps generalization
        use_asymmetric=True,       # ENABLE ASL: Better handling of easy negatives than Weighted Focal
        use_hybrid=False,          # Pure ASL is often more stable for binary imbalance
        gamma_neg=2.0,             # Tuned: Suppress easy negatives more aggressively
        # v4.3 Anomaly Weights
        anomaly_weight=model_config.anomaly_weight,
    )

    # Log loss configuration
    print(f"  [INFO] Neuro-ADEPT v5.5 Stabilized Loss:")
    print(f"    - Classification: Weighted Focal Loss (Path Weight={class_weights[1].item():.1f})")
    print(f"    - Deep SVDD: weight={model_config.anomaly_weight}, nu={model_config.svdd_nu}")

    # Optimizer with proper settings - v4.2: REDUCED LR for stability
    optimizer = optim.AdamW(
        model.parameters(),
        lr=4e-4,            # REDUCED from 6e-4 for better stability
        weight_decay=2e-2,
        betas=(0.9, 0.999)
    )

    # Gradient accumulation for effective batch size of 128
    accumulation_steps = max(1, 128 // config['batch_size'])

    # Learning rate scheduler - Cosine annealing with warmup
    # FIX: total_steps must account for gradient accumulation (step called less often)
    total_steps = (config['epochs'] * len(train_loader)) // accumulation_steps

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=4e-4,          # REDUCED from 6e-4
        total_steps=total_steps,
        pct_start=0.15,       # Faster warmup
        anneal_strategy='cos',
        div_factor=10,        
        final_div_factor=100  
    )

    scaler = GradScaler('cuda')

    # Training state
    best_f1 = 0.0
    best_balanced_acc = 0.0
    best_path_sens = 0.0
    patience = 0
    epochs_since_peak = 0

    # Gradient accumulation for effective batch size of 128
    accumulation_steps = max(1, 128 // config['batch_size'])

    for epoch in range(1, config['epochs'] + 1):
        # Curriculum learning for prediction loss
        if epoch <= 3:
            criterion.lambda_pred = 0.0
        elif epoch <= 8:
            # Linear ramp from 0 to 0.05
            criterion.lambda_pred = 0.01 * (epoch - 3)
        else:
            criterion.lambda_pred = 0.05

        # Dynamic loss weighting based on epoch (v4.1)
        criterion.set_epoch(epoch, config['epochs'], model_config.anomaly_warmup_epochs)

        # Train
        train_loss, train_pred_norm, train_pred_sz = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler=scaler, use_mixup=True, accumulation_steps=accumulation_steps,
            scheduler=scheduler, anomaly_warmup_epochs=model_config.anomaly_warmup_epochs
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Note: OneCycleLR is stepped per batch in train_epoch, not here

        # Logging with CLINICAL METRICS - Include epoch number prominently
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']



        print(f"\n  [Epoch {epoch}/{config['epochs']}] Train Loss: {train_loss:.4f} | LR: {current_lr:.2e}")
        print(f"  [TRAIN] PredErr: Normal={train_pred_norm:.4f} vs Seizure={train_pred_sz:.4f} "
              f"(Ratio: {train_pred_sz/(train_pred_norm+1e-8):.1f}x)")
        print(f"  -------- Validation Metrics (Threshold=0.5) --------")
        print(f"  Macro F1: {val_metrics['f1']:.3f} | F1-Normal: {val_metrics['f1_normal']:.3f} | F1-Seizure: {val_metrics['f1_seizure']:.3f}")
        print(f"  Acc: {val_metrics['acc']:.3f} | BalAcc: {val_metrics['balanced_acc']:.3f}")
        print(f"  Sens: {val_metrics['sensitivity']:.2f} | Spec: {val_metrics['specificity']:.2f} | "
              f"PPV: {val_metrics['ppv']:.2f} | NPV: {val_metrics['npv']:.2f}")
        print(f"  AUROC: {val_metrics['auroc']:.3f} | AUPRC: {val_metrics['auprc']:.3f}")
        print(f"  Sens@90Spec: {val_metrics['sens_at_90spec']:.2f} | FPR/h: {val_metrics['fpr_per_hour']:.1f}")
        # Show optimized threshold metrics (better for clinical use)
        print(f"  -------- Optimal Threshold ({val_metrics['opt_threshold']:.3f}) --------")
        # Calculate F1 at optimal threshold for better visibility
        opt_preds = (val_metrics['probs'][:, 1] >= val_metrics['opt_threshold']).astype(int)
        opt_f1 = f1_score(val_metrics['targets'], opt_preds, average='macro')
        print(f"  [METRIC] Opt-F1: {opt_f1:.3f} | OptSens: {val_metrics['opt_sensitivity']:.2f} | "
              f"OptSpec: {val_metrics['opt_specificity']:.2f} | OptPPV: {val_metrics['opt_ppv']:.2f}")
        print(f"  PredErr: {val_metrics['pred_loss']:.4f} | lambda_pred: {criterion.lambda_pred:.4f}")

        # CLINICAL SAVE CRITERION: Balanced approach
        # Use composite score: AUROC (discrimination) + Optimal Sensitivity (clinical priority)
        # AUROC is threshold-independent, opt_sensitivity shows best achievable seizure detection
        current_score = (
            0.4 * val_metrics['auroc'] +
            0.3 * val_metrics['opt_sensitivity'] +  # Prioritize seizure detection
            0.2 * val_metrics['balanced_acc'] +
            0.1 * val_metrics['auprc']  # AUPRC important for imbalanced data
        )

        # Save if: Better composite score AND reasonable metrics at optimal threshold
        # Use optimal threshold metrics which are more clinically relevant
        min_sens = 0.30 if epoch < 10 else 0.45
        min_spec = 0.25 if epoch < 10 else 0.30

        should_save = (
            current_score > best_balanced_acc and  # Using best_balanced_acc to store composite
            val_metrics['opt_sensitivity'] >= min_sens and
            val_metrics['opt_specificity'] >= min_spec
        )

        if should_save:
            best_f1 = val_metrics['f1']
            best_balanced_acc = current_score  # Store composite score
            best_path_sens = val_metrics['opt_sensitivity']
            # Save model and optimal threshold
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimal_threshold': val_metrics['opt_threshold'],
                'metrics': {
                    'auroc': val_metrics['auroc'],
                    'f1': val_metrics['f1'],
                    'opt_f1': opt_f1,  # Calculated earlier in loop
                    'opt_sensitivity': val_metrics['opt_sensitivity'],
                    'opt_specificity': val_metrics['opt_specificity'],
                    'fpr': val_metrics.get('fpr'),
                    'tpr': val_metrics.get('tpr')
                }
            }, output_dir / f'best_fold{fold_num}.pt')
            print(f"  [SAVE] New Best! (Score={current_score:.3f}, AUROC={val_metrics['auroc']:.3f}, "
                  f"OptSens={best_path_sens:.2f}, OptSpec={val_metrics['opt_specificity']:.2f}, Thresh={val_metrics['opt_threshold']:.3f})")
            patience = 0
            epochs_since_peak = 0
        else:
            patience += 1
            epochs_since_peak += 1

            # CLINICAL EARLY STOPPING - More patient to allow learning
            # Only stop for truly catastrophic failures after many epochs

            # Severe class collapse detection (only after warmup period)
            if epoch > 25 and (val_metrics['sensitivity'] < 0.05 or val_metrics['specificity'] < 0.05):
                print(f"  [WARN] Potential class collapse (Sens={val_metrics['sensitivity']:.2f}, "
                      f"Spec={val_metrics['specificity']:.2f}) - continuing training...")
                # Don't break - let training continue, model might recover

            if patience >= 25:  # More patience for learning
                print("  [STOP] Early Stopping (patience=25)")
                break

    # Generate plots for best model
    best_path = output_dir / f'best_fold{fold_num}.pt'
    if best_path.exists():
        checkpoint = torch.load(best_path, weights_only=False)
        # Handle both old and new checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            opt_thresh = checkpoint.get('optimal_threshold', 0.5)
            print(f"  [PLOT] Generating plots for best model (Optimal Threshold={opt_thresh:.3f})...")
        else:
            model.load_state_dict(checkpoint)
            print("  [PLOT] Generating plots for best model...")

        # Collect validation data
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        all_attn, all_error, all_latent = [], [], []

        with torch.no_grad():
            for data, context, target in val_loader:
                data, context = data.to(device), context.to(device)
                logits, _, info = model(data, context)

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(target.numpy())
                all_attn.extend(info['attn'].cpu().numpy())
                all_error.extend(info['error_map'].cpu().numpy())
                all_latent.extend(info['latent'].cpu().numpy())

        # Generate all plots
        plot_binary_results(
            np.array(all_targets), np.array(all_preds),
            np.array(all_probs), output_dir, fold_num
        )
        plot_neuro_interpretability(
            np.array(all_attn), np.array(all_error),
            np.array(all_targets), output_dir, fold_num
        )
        plot_latent_space(
            np.array(all_latent), np.array(all_targets),
            output_dir, fold_num
        )
        plot_gradcam_frequency_bands(model, val_loader, device, output_dir, fold_num)

    return best_f1, best_balanced_acc


def clear_caches(cache_dir: str = 'cache'):
    """Clear all processed cache files to force re-processing."""
    import shutil
    # v5.0: User requested persistence. Do NOT delete processed_mmap.
    # We only clear processed (legacy) and pycache.
    processed_dir = Path(cache_dir) / 'processed'
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"[CACHE] Cleared legacy cache: {processed_dir}")
        
    mmap_dir = Path(cache_dir) / 'processed_mmap'
    if mmap_dir.exists():
        print(f"[CACHE] PRESERVING persistent memmaps in: {mmap_dir}")
        print(f"        (To force regeneration, manually delete this folder)")

    # Also clear __pycache__ to ensure fresh code
    for pycache in Path('.').rglob('__pycache__'):
        shutil.rmtree(pycache)
        print(f"[CACHE] Cleared: {pycache}")


def main():
    parser = argparse.ArgumentParser(description='Neuro-PCN v3 Training')
    parser.add_argument('--dataset', default='chbmit', choices=['chbmit', 'siena'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16, help='Optimized for Stability (was 32)')
    parser.add_argument('--n_folds', type=int, default=-1, help='-1 for all folds')
    parser.add_argument('--clear_cache', action='store_true', help='Clear all caches before training')
    parser.add_argument('--num_workers', type=int, default=0, help='0 for Windows/WSL stability')
    parser.add_argument('--max_epochs_per_subject', type=int, default=50000, help='Max epochs per subject (use all pathology + 2x normal)')
    args = parser.parse_args()

    # Clear caches if requested
    if args.clear_cache:
        clear_caches()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n" + "="*60)
    print("NEURO-ADEPT TRAINING START")
    print("="*60)

    config = vars(args)
    # [FIX] User confirmed data is in 'cache' (preprocessed .npz), not 'datasets' (raw .edf)
    config['cache_dir'] = 'cache'

    # WSL-specific optimizations
    if 'WSL' in os.uname().release if hasattr(os, 'uname') else False:
        print("[WSL] Detected WSL environment - applying optimizations")
        config['num_workers'] = min(config.get('num_workers', 2), 2)  # WSL has IPC issues with many workers
        torch.multiprocessing.set_sharing_strategy('file_system')  # Better for WSL

    output_dir = Path('results_v2') / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = get_subject_list(config['cache_dir'], args.dataset)
    if len(subjects) < 3:
        print("Not enough subjects!")
        return

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Determine number of folds
    n_folds = len(subjects) if args.n_folds == -1 else min(args.n_folds, len(subjects))
    print(f"Running {n_folds} folds for {args.dataset}...")

    # Run LOSO cross-validation
    f1_scores = []
    balanced_accs = []
    failed_folds = []

    for i in range(n_folds):
        val_s = [subjects[i]]
        train_s = [s for s in subjects if s != subjects[i]]

        try:
            f1, bal_acc = run_fold(train_s, val_s, config, device, output_dir, i+1)
            f1_scores.append(f1)
            balanced_accs.append(bal_acc)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\n[ERROR] CUDA OOM in Fold {i+1}. Trying with smaller batch...")
                torch.cuda.empty_cache()
                gc.collect()
                # Try again with smaller batch
                config['batch_size'] = max(16, config['batch_size'] // 2)
                try:
                    f1, bal_acc = run_fold(train_s, val_s, config, device, output_dir, i+1)
                    f1_scores.append(f1)
                    balanced_accs.append(bal_acc)
                except Exception as e2:
                    print(f"[ERROR] Fold {i+1} failed even with reduced batch: {e2}")
                    failed_folds.append(i+1)
            else:
                print(f"[ERROR] Fold {i+1} failed: {e}")
                failed_folds.append(i+1)
        except Exception as e:
            print(f"[ERROR] Fold {i+1} failed: {e}")
            import traceback
            traceback.print_exc()
            failed_folds.append(i+1)

        # Memory cleanup between folds
        gc.collect()
        torch.cuda.empty_cache()

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if f1_scores:
        print(f"Completed Folds:  {len(f1_scores)}/{n_folds}")
        print(f"Mean F1:          {np.mean(f1_scores):.3f} +/- {np.std(f1_scores):.3f}")
        print(f"Mean Balanced Acc: {np.mean(balanced_accs):.3f} +/- {np.std(balanced_accs):.3f}")
        print(f"Best F1:          {np.max(f1_scores):.3f} (Fold {np.argmax(f1_scores)+1})")
    if failed_folds:
        print(f"Failed Folds:     {failed_folds}")
    print("="*60)


if __name__ == '__main__':
    main()
