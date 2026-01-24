#!/usr/bin/env python3
"""
Neuro-ADEPT Preprocessing Pipeline

Converts raw CHB-MIT and Siena .edf files to preprocessed .npz format.
Handles seizure annotations and extracts 5-second epochs.

Usage:
    python preprocess.py --dataset chbmit
    python preprocess.py --dataset siena
    python preprocess.py --all
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import mne for EDF reading
try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("[ERROR] MNE library not found. Install with: pip install mne")
    sys.exit(1)


def parse_seizure_annotations(summary_file: Path) -> dict:
    """
    Parse CHB-MIT style seizure annotations from summary file.
    
    Returns dict: {filename: [(start_sec, end_sec), ...]}
    """
    seizures = {}
    current_file = None
    
    if not summary_file.exists():
        return seizures
    
    with open(summary_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()
            seizures[current_file] = []
        elif line.startswith('Seizure') and 'Start Time:' in line:
            try:
                start = int(line.split(':')[1].strip().split()[0])
                # Find corresponding end time
            except:
                pass
        elif line.startswith('Seizure') and 'End Time:' in line:
            try:
                end = int(line.split(':')[1].strip().split()[0])
                if current_file and seizures[current_file] is not None:
                    # Pair with last start if available
                    pass
            except:
                pass
    
    return seizures


def parse_chb_seizures(subject_dir: Path) -> dict:
    """
    Parse seizure annotations for CHB-MIT subject.
    Returns {edf_file: [(start_sec, end_sec), ...]}
    """
    seizures = {}
    summary_file = list(subject_dir.glob('*-summary.txt'))
    
    if not summary_file:
        return seizures
    
    summary_file = summary_file[0]
    
    with open(summary_file, 'r') as f:
        content = f.read()
    
    # Parse file-by-file
    current_file = None
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()
            seizures[current_file] = []
            
        elif 'Number of Seizures' in line:
            try:
                num_seizures = int(line.split(':')[1].strip().split()[0])
                # Read next seizure entries
                for j in range(num_seizures):
                    # Find start and end times
                    while i < len(lines) - 1:
                        i += 1
                        l = lines[i].strip()
                        if 'Seizure' in l and 'Start' in l:
                            start = int(l.split(':')[1].strip().split()[0])
                            i += 1
                            end_line = lines[i].strip()
                            end = int(end_line.split(':')[1].strip().split()[0])
                            if current_file:
                                seizures[current_file].append((start, end))
                            break
            except Exception as e:
                pass
        
        i += 1
    
    return seizures


def preprocess_edf(edf_path: Path, seizure_times: list, epoch_len: float = 5.0,
                   sample_rate: int = 256, target_channels: int = 19) -> tuple:
    """
    Preprocess a single EDF file into epochs.
    Returns:
        X: (n_epochs, channels, samples) float32
        y: (n_epochs,) int64 - 0=interictal, 1=ictal
    """
    import gc
    try:
        # Load EDF - use preload=False to save memory initially if MNE supports it
        # But for reliability with different MNE versions, we load but optimize immediately
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Resample if needed
        if int(raw.info['sfreq']) != sample_rate:
            raw = raw.resample(sample_rate, n_jobs=1)  # Single core to save memory overhead
        
        # Get data - immediately convert to float32 to save 50% RAM
        # MNE returns float64 by default
        data = raw.get_data().astype(np.float32)
        del raw
        gc.collect()
        
        n_channels, n_samples = data.shape
        
        # Standardize channels
        if n_channels > target_channels:
            data = data[:target_channels, :]
        elif n_channels < target_channels:
            # Pad with zeros
            pad = np.zeros((target_channels - n_channels, n_samples), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
            
        # Create epochs
        samples_per_epoch = int(epoch_len * sample_rate)
        n_epochs = n_samples // samples_per_epoch
        
        if n_epochs == 0:
            return None, None
        
        # Truncate to complete epochs
        data = data[:, :n_epochs * samples_per_epoch]
        
        # Reshape to epochs: (channels, n_epochs, samples) -> (n_epochs, channels, samples)
        # We do this carefully to avoid huge copies
        X = data.reshape(target_channels, n_epochs, samples_per_epoch)
        X = X.transpose(1, 0, 2).copy() # Copy to enforce contiguous memory
        
        del data
        gc.collect()
        
        # Create labels
        y = np.zeros(n_epochs, dtype=np.int64)
        
        for start_sec, end_sec in seizure_times:
            start_epoch = int(start_sec / epoch_len)
            end_epoch = int(np.ceil(end_sec / epoch_len))
            
            start_epoch = max(0, min(start_epoch, n_epochs - 1))
            end_epoch = max(0, min(end_epoch, n_epochs))
            
            y[start_epoch:end_epoch] = 1  # Ictal
            
        return X, y
        
    except Exception as e:
        print(f"    [WARN] Failed to process {edf_path.name}: {e}")
        return None, None


def preprocess_subject(subject_dir: Path, output_dir: Path, dataset: str = 'chbmit'):
    """
    Preprocess all EDF files for a single subject using incremental writing.
    Low RAM version.
    """
    import tempfile
    import shutil
    import gc
    
    subject_name = subject_dir.name
    output_file = output_dir / f"{subject_name}.npz"
    
    if output_file.exists():
        print(f"  [SKIP] {subject_name} already preprocessed")
        return True
    
    print(f"  [PROCESS] {subject_name}...")
    
    # Get seizure annotations
    seizures = parse_chb_seizures(subject_dir)
    
    # Find all EDF files
    edf_files = sorted(subject_dir.glob('*.edf'))
    if not edf_files:
        print(f"    [WARN] No EDF files found in {subject_name}")
        return False
        
    # Use a temporary directory for incremental storage
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_data_file = temp_dir_path / "data.bin"
        
        all_y = []
        total_epochs = 0
        samples_per_epoch = int(5.0 * 256)
        n_channels = 19
        
        # Open binary file for appending
        with open(temp_data_file, 'wb') as f_out:
            for edf_file in edf_files:
                seizure_times = seizures.get(edf_file.name, [])
                
                # Process single file
                X_chunk, y_chunk = preprocess_edf(edf_file, seizure_times)
                
                if X_chunk is not None:
                    # Write X immediately to disk
                    X_chunk.tofile(f_out)
                    
                    # Keep y in memory (it's small)
                    all_y.append(y_chunk)
                    total_epochs += len(y_chunk)
                    
                    # Cleanup
                    del X_chunk, y_chunk
                    gc.collect()
        
        if total_epochs == 0:
            print(f"    [WARN] No valid epochs extracted for {subject_name}")
            return False
            
        # Concatenate labels
        y_all = np.concatenate(all_y, axis=0)
        
        # Check integrity
        expected_bytes = total_epochs * n_channels * samples_per_epoch * 4 # float32 = 4 bytes
        actual_bytes = temp_data_file.stat().st_size
        
        if actual_bytes != expected_bytes:
            print(f"    [ERROR] Size mismatch. Expected {expected_bytes}, got {actual_bytes}")
            return False
            
        # Memory map the binary file to save as NPZ without loading to RAM
        X_mmap = np.memmap(temp_data_file, dtype='float32', mode='r', 
                           shape=(total_epochs, n_channels, samples_per_epoch))
                           
        # Verify shape
        # print(f"    Writing .npz (Shape: {X_mmap.shape})...")
        
        # Save compressed (reads from disk chunk by chunk primarily)
        np.savez_compressed(output_file, X=X_mmap, y=y_all)
        
        # Cleanup mmap
        del X_mmap
        # Temp dir auto-cleans up
    
    n_seizure = (y_all > 0).sum()
    print(f"    Saved: {len(y_all)} epochs ({n_seizure} seizure, {len(y_all)-n_seizure} normal)")
    
    return True


def preprocess_siena_subject(subject_dir: Path, output_dir: Path):
    """
    Preprocess Siena dataset subject using incremental writing.
    Low RAM version.
    """
    import tempfile
    import shutil
    import gc

    subject_name = subject_dir.name
    output_file = output_dir / f"{subject_name}.npz"
    
    if output_file.exists():
        print(f"  [SKIP] {subject_name} already preprocessed")
        return True
    
    print(f"  [PROCESS] {subject_name}...")
    
    # Find all EDF files
    edf_files = sorted(subject_dir.glob('*.edf'))
    if not edf_files:
        print(f"    [WARN] No EDF files found in {subject_name}")
        return False
    
    # Use a temporary directory for incremental storage
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_data_file = temp_dir_path / "data.bin"
        
        all_y = []
        total_epochs = 0
        samples_per_epoch = int(5.0 * 256)
        n_channels = 19
        
        # Open binary file for appending
        with open(temp_data_file, 'wb') as f_out:
            for edf_file in edf_files:
                # Siena: check for .seizures file
                seizure_file = Path(str(edf_file) + '.seizures')
                seizure_times = []
                
                if seizure_file.exists():
                    try:
                        with open(seizure_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 2:
                                    start = float(parts[0])
                                    end = float(parts[1])
                                    seizure_times.append((start, end))
                    except:
                        pass
                
                # Process single file
                X_chunk, y_chunk = preprocess_edf(edf_file, seizure_times)
                
                if X_chunk is not None:
                    # Write X immediately to disk
                    X_chunk.tofile(f_out)
                    
                    # Keep y in memory
                    all_y.append(y_chunk)
                    total_epochs += len(y_chunk)
                    
                    # Cleanup
                    del X_chunk, y_chunk
                    gc.collect()
    
        if total_epochs == 0:
            print(f"    [WARN] No valid epochs extracted for {subject_name}")
            return False
        
        # Concatenate labels
        y_all = np.concatenate(all_y, axis=0)
        
        # Check integrity
        expected_bytes = total_epochs * n_channels * samples_per_epoch * 4
        actual_bytes = temp_data_file.stat().st_size
        
        if actual_bytes != expected_bytes:
            print(f"    [ERROR] Size mismatch. Expected {expected_bytes}, got {actual_bytes}")
            return False
        
        # Memory map the binary file
        X_mmap = np.memmap(temp_data_file, dtype='float32', mode='r', 
                           shape=(total_epochs, n_channels, samples_per_epoch))
        
        # Save compressed
        np.savez_compressed(output_file, X=X_mmap, y=y_all)
        
        # Cleanup
        del X_mmap

    n_seizure = (y_all > 0).sum()
    print(f"    Saved: {len(y_all)} epochs ({n_seizure} seizure, {len(y_all)-n_seizure} normal)")
    
    return True



def preprocess_dataset_parallel(dataset: str, data_dir: Path, cache_dir: Path, workers: int = 4):
    """
    Preprocess dataset in parallel.
    """
    input_dir = data_dir / dataset
    output_dir = cache_dir / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        print(f"[ERROR] Dataset directory not found: {input_dir}")
        return False
    
    # Find all subject directories
    if dataset == 'chbmit':
        subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('chb')])
    elif dataset == 'siena':
        subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    else:
        subject_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    
    if not subject_dirs:
        print(f"[ERROR] No subject directories found in {input_dir}")
        return False
    
    print(f"\n{'='*60}")
    print(f"PREPROCESSING: {dataset.upper()} (Parallel: {workers} workers)")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Subjects: {len(subject_dirs)}")
    print(f"{'='*60}\n")
    
    # Prepare arguments for map
    # We need a wrapper because pool.map takes one arg
    from functools import partial
    import multiprocessing
    
    success = 0
    
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            if dataset == 'siena':
                # Siena
                func = partial(preprocess_siena_subject, output_dir=output_dir)
                results = list(tqdm(pool.imap(func, subject_dirs), total=len(subject_dirs)))
                success = sum(results)
            else:
                # CHB-MIT
                func = partial(preprocess_subject, output_dir=output_dir, dataset=dataset)
                results = list(tqdm(pool.imap(func, subject_dirs), total=len(subject_dirs)))
                success = sum(results)
    else:
        # Serial fallback
        from tqdm import tqdm
        for subject_dir in tqdm(subject_dirs):
            if dataset == 'siena':
                if preprocess_siena_subject(subject_dir, output_dir):
                    success += 1
            else:
                if preprocess_subject(subject_dir, output_dir, dataset):
                    success += 1
    
    print(f"\n[DONE] Preprocessed {success}/{len(subject_dirs)} subjects for {dataset}")
    return success > 0


def main():
    parser = argparse.ArgumentParser(description='Preprocess EEG datasets')
    parser.add_argument('--dataset', choices=['chbmit', 'siena'], default='chbmit',
                        help='Dataset to preprocess')
    parser.add_argument('--all', action='store_true', help='Preprocess all datasets')
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help='Number of parallel workers')
    parser.add_argument('--data_dir', default='datasets', help='Raw data directory')
    parser.add_argument('--cache_dir', default='cache', help='Output cache directory')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    
    if args.all:
        preprocess_dataset_parallel('chbmit', data_dir, cache_dir, args.workers)
        preprocess_dataset_parallel('siena', data_dir, cache_dir, args.workers)
    else:
        preprocess_dataset_parallel(args.dataset, data_dir, cache_dir, args.workers)


if __name__ == '__main__':
    from tqdm import tqdm 
    main()

