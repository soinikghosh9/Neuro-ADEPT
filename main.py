#!/usr/bin/env python3
"""
Neuro-ADEPT v4.4: End-to-End Seizure Detection Pipeline

This script runs the complete workflow:
1. Preprocess raw EDF files to NPZ cache (if needed)
2. Train model on CHB-MIT and/or Siena datasets
3. Export all results to organized output directory

Usage:
    python main.py                          # Run both datasets
    python main.py --dataset chbmit         # Run only CHB-MIT
    python main.py --dataset siena          # Run only Siena
    python main.py --skip_preprocess        # Skip preprocessing (use existing cache)
    python main.py --epochs 50 --n_folds -1 # Full training

Output:
    results/
    ├── chbmit/
    │   ├── fold_*/                         # Per-fold results
    │   ├── summary.json                    # Aggregate metrics
    │   └── training_log.txt                # Training output
    └── siena/
        └── ...
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def run_preprocessing(datasets: list, data_dir: str = 'datasets', cache_dir: str = 'cache'):
    """
    Run preprocessing for specified datasets.
    """
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING")
    print("="*70)
    
    for dataset in datasets:
        cache_path = Path(cache_dir) / dataset
        if cache_path.exists() and list(cache_path.glob('*.npz')):
            n_subjects = len(list(cache_path.glob('*.npz')))
            print(f"[{dataset.upper()}] Found {n_subjects} cached subjects. Skipping preprocessing.")
            continue
        
        print(f"\n[{dataset.upper()}] Running preprocessing...")
        cmd = [
            sys.executable, 'preprocess.py',
            '--dataset', dataset,
            '--data_dir', data_dir,
            '--cache_dir', cache_dir
        ]
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"[ERROR] Preprocessing failed for {dataset}")
            return False
    
    return True


def run_training(dataset: str, epochs: int, batch_size: int, n_folds: int,
                 output_dir: Path, log_file: Path):
    """
    Run training for a single dataset and capture output.
    """
    print(f"\n[{dataset.upper()}] Starting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Folds: {'All' if n_folds == -1 else n_folds}")
    print(f"  Output: {output_dir}")
    
    cmd = [
        sys.executable, '-u', 'train.py',
        '--dataset', dataset,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--n_folds', str(n_folds)
    ]
    
    # Run and capture output
    with open(log_file, 'w', encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        for line in process.stdout:
            print(line, end='')  # Print to console
            f.write(line)        # Write to log file
        
        process.wait()
    
    return process.returncode == 0


def collect_results(results_dir: Path, dataset: str) -> dict:
    """
    Collect and aggregate results from training.
    """
    dataset_dir = results_dir / dataset
    if not dataset_dir.exists():
        return {}
    
    # Find all fold results
    fold_dirs = sorted(dataset_dir.glob('*.pt'))
    
    results = {
        'dataset': dataset,
        'timestamp': datetime.now().isoformat(),
        'n_folds': len(fold_dirs),
        'fold_results': [],
        'aggregate': {}
    }
    
    # Try to parse metrics from saved models
    import torch
    
    all_auroc = []
    all_f1 = []
    all_sens = []
    all_spec = []
    
    for fold_path in fold_dirs:
        try:
            checkpoint = torch.load(fold_path, weights_only=False, map_location='cpu')
            if isinstance(checkpoint, dict) and 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                results['fold_results'].append({
                    'fold': fold_path.stem,
                    'threshold': checkpoint.get('optimal_threshold', 0.5),
                    **metrics
                })
                all_auroc.append(metrics.get('auroc', 0))
                all_f1.append(metrics.get('opt_f1', 0))
                all_sens.append(metrics.get('opt_sensitivity', 0))
                all_spec.append(metrics.get('opt_specificity', 0))
                
                # Check for curve data
                if 'fpr' in metrics and 'tpr' in metrics:
                    results['fold_results'][-1]['fpr'] = metrics['fpr']
                    results['fold_results'][-1]['tpr'] = metrics['tpr']
        except Exception as e:
            pass
    
    # Compute aggregates
    if all_auroc:
        import numpy as np
        results['aggregate'] = {
            'mean_auroc': float(np.mean(all_auroc)),
            'std_auroc': float(np.std(all_auroc)),
            'mean_sensitivity': float(np.mean(all_sens)),
            'std_sensitivity': float(np.std(all_sens)),
            'mean_specificity': float(np.mean(all_spec)),
            'std_specificity': float(np.std(all_spec)),
            'mean_f1': float(np.mean(all_f1)),
            'std_f1': float(np.std(all_f1)),
        }
    
    return results


def export_results(results: dict, output_dir: Path, dataset: str):
    """
    Export results to JSON and markdown summary.
    """
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    json_path = dataset_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")
    
    # Save Markdown summary
    md_path = dataset_dir / 'summary.md'
    with open(md_path, 'w') as f:
        f.write(f"# Neuro-ADEPT v4.4 Results: {dataset.upper()}\n\n")
        f.write(f"**Generated**: {results.get('timestamp', 'N/A')}\n\n")
        
        if results.get('aggregate'):
            agg = results['aggregate']
            f.write("## Aggregate Metrics\n\n")
            f.write("| Metric | Mean | Std |\n")
            f.write("|--------|------|-----|\n")
            f.write(f"| AUROC | {agg.get('mean_auroc', 0):.3f} | {agg.get('std_auroc', 0):.3f} |\n")
            f.write(f"| Opt-F1 | {agg.get('mean_f1', 0):.3f} | {agg.get('std_f1', 0):.3f} |\n")
            f.write(f"| Sensitivity | {agg.get('mean_sensitivity', 0):.3f} | {agg.get('std_sensitivity', 0):.3f} |\n")
            f.write(f"| Specificity | {agg.get('mean_specificity', 0):.3f} | {agg.get('std_specificity', 0):.3f} |\n")
            f.write("\n")
        
        if results.get('fold_results'):
            f.write("## Per-Fold Results\n\n")
            f.write("| Fold | AUROC | Opt-F1 | Sensitivity | Specificity | Threshold |\n")
            f.write("|------|-------|--------|-------------|-------------|----------|\n")
            for fold in results['fold_results']:
                f.write(f"| {fold.get('fold', 'N/A')} | "
                       f"{fold.get('auroc', 0):.3f} | "
                       f"{fold.get('opt_f1', 0):.3f} | "
                       f"{fold.get('opt_sensitivity', 0):.3f} | "
                       f"{fold.get('opt_specificity', 0):.3f} | "
                       f"{fold.get('threshold', 0.5):.3f} |\n")
    
    print(f"  Saved: {md_path}")
    
    # Generate Aggregate Plots
    generate_aggregate_plots(results, dataset_dir)


def generate_aggregate_plots(results: dict, output_dir: Path):
    """
    Generate aggregate ROC and other visual summaries.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    folds_with_curves = [f for f in results['fold_results'] if 'fpr' in f and f['fpr'] is not None]
    
    if not folds_with_curves:
        print("  [WARN] No curve data found for aggregate plots")
        return

    # ==========================================
    # AGGREGATE ROC CURVE
    # ==========================================
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for fold in folds_with_curves:
        fpr = fold['fpr']
        tpr = fold['tpr']
        # Interpolate TPR to mean FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(fold['auroc'])
        ax.plot(fpr, tpr, color='gray', alpha=0.1, linewidth=0.5)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    ax.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})', lw=2, alpha=0.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='b', alpha=0.2, label='$\pm$ 1 std. dev.')

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=0.8)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(f'Aggregate ROC Curve - {results["dataset"].upper()}', fontweight='bold')
    ax.legend(loc="lower right")
    
    save_path = output_dir / 'aggregate_roc.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def clear_pycache():
    """
    Clear only __pycache__ directories (not NPZ data cache).
    """
    import shutil
    
    cleared = []
    # Recursively find all __pycache__ directories
    for p in Path('.').rglob('__pycache__'):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            cleared.append(str(p))
            
    if cleared:
        print(f"[CACHE] Cleared {len(cleared)} __pycache__ directories.")
    else:
        print("[CACHE] No __pycache__ directories found")


def print_banner():
    """Print startup banner."""
    print("\n" + "="*70)
    print("  NEURO-ADEPT v4.5: Dual-Stream Anomaly Detection for EEG Seizures")
    print("  Brain-Inspired Framework for Interpretable Seizure Detection")
    print("="*70)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Neuro-ADEPT End-to-End Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                              # Both datasets, default settings
    python main.py --dataset chbmit --epochs 50 # CHB-MIT only, 50 epochs
    python main.py --skip_preprocess            # Use existing cache
    python main.py --quick                      # Quick test (5 epochs, 1 fold)
    python main.py --clear_cache                # Clear pycache before running
        """
    )
    
    parser.add_argument('--dataset', choices=['chbmit', 'siena', 'both'], default='both',
                        help='Dataset to process')
    parser.add_argument('--epochs', type=int, default=25, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--n_folds', type=int, default=-1, help='Number of folds (-1 for all)')
    parser.add_argument('--skip_preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (5 epochs, 1 fold)')
    parser.add_argument('--clear_cache', action='store_true', 
                        help='Clear __pycache__ directories only (not NPZ data cache)')
    parser.add_argument('--data_dir', default='datasets', help='Raw data directory')
    parser.add_argument('--cache_dir', default='cache', help='Preprocessed data cache')
    parser.add_argument('--output_dir', default='results', help='Results output directory')
    
    args = parser.parse_args()
    
    # Clear pycache if requested
    if args.clear_cache:
        clear_pycache()
    
    # Quick mode overrides
    if args.quick:
        args.epochs = 5
        args.n_folds = 1
    
    print_banner()
    
    # Determine datasets to process
    if args.dataset == 'both':
        datasets = ['chbmit', 'siena']
    else:
        datasets = [args.dataset]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Preprocessing
    if not args.skip_preprocess:
        if not run_preprocessing(datasets, args.data_dir, args.cache_dir):
            print("[ERROR] Preprocessing failed. Exiting.")
            return 1
    
    # Step 2: Training
    print("\n" + "="*70)
    print("STEP 2: TRAINING")
    print("="*70)
    
    all_results = {}
    
    for dataset in datasets:
        log_file = output_dir / dataset / 'training_log.txt'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        success = run_training(
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_folds=args.n_folds,
            output_dir=output_dir / dataset,
            log_file=log_file
        )
        
        if not success:
            print(f"[WARN] Training failed for {dataset}")
            continue
        
        # Collect results from results_v2 (where train.py saves models)
        results = collect_results(Path('results_v2'), dataset)
        all_results[dataset] = results
        
        # Export
        export_results(results, output_dir, dataset)
    
    # Step 3: Final Summary
    print("\n" + "="*70)
    print("STEP 3: FINAL SUMMARY")
    print("="*70)
    
    for dataset, results in all_results.items():
        print(f"\n[{dataset.upper()}]")
        if results.get('aggregate'):
            agg = results['aggregate']
            print(f"  AUROC:       {agg.get('mean_auroc', 0):.3f} ± {agg.get('std_auroc', 0):.3f}")
            print(f"  Opt-F1:      {agg.get('mean_f1', 0):.3f} ± {agg.get('std_f1', 0):.3f}")
            print(f"  Sensitivity: {agg.get('mean_sensitivity', 0):.3f} ± {agg.get('std_sensitivity', 0):.3f}")
            print(f"  Specificity: {agg.get('mean_specificity', 0):.3f} ± {agg.get('std_specificity', 0):.3f}")
        else:
            print("  No results available")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print(f"Results saved to: {output_dir.absolute()}")
    print("="*70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
