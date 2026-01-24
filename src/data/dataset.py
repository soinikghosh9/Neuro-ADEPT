#!/usr/bin/env python3
"""
Optimized EEG Dataset for LightPCN (v4.4 - Balanced Sampling)

Improvements:
1. Memmap Caching: Uses numpy memory mapping for zero-RAM data loading
2. Unlimited Epochs: Loads standard full dataset without OOM
3. Float16 Storage: efficient disk usage
4. BalancedBatchSampler: Creates balanced batches without double-penalty
5. Temporal-Aware Augmentation: Safe augmentation for seizure samples
"""

import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import gc
import shutil


class BalancedBatchSampler(Sampler):
    """
    Creates balanced training batches for imbalanced seizure detection.
    
    CRITICAL FIX for F1-Seizure stagnation:
    - Each batch contains ~minority_ratio seizure samples
    - Normal samples are randomly subsampled each epoch
    - Avoids the double-penalty of WeightedRandomSampler + class weights
    
    Clinical Rationale:
    - Ensures model sees sufficient seizure examples each batch
    - Maintains more realistic class ratios than full oversampling
    - Works WITH moderate class weights, not against them
    """
    
    def __init__(self, labels: torch.Tensor, batch_size: int, 
                 minority_ratio: float = 0.4, drop_last: bool = True):
        """
        Args:
            labels: Full label tensor (0=normal, 1=seizure)
            batch_size: Samples per batch
            minority_ratio: Fraction of seizure samples per batch (default 0.4 = 40%)
            drop_last: Drop incomplete final batch
        """
        super().__init__(None)
        self.labels = labels
        self.batch_size = batch_size
        self.minority_ratio = minority_ratio
        self.drop_last = drop_last
        
        # Separate indices by class
        self.normal_indices = (labels == 0).nonzero(as_tuple=True)[0].tolist()
        self.seizure_indices = (labels == 1).nonzero(as_tuple=True)[0].tolist()
        
        # Samples per class per batch
        self.seizure_per_batch = max(1, int(batch_size * minority_ratio))
        self.normal_per_batch = batch_size - self.seizure_per_batch
        
        # Calculate number of batches (limited by seizure samples)
        if len(self.seizure_indices) > 0:
            self.n_batches = len(self.seizure_indices) // self.seizure_per_batch
        else:
            self.n_batches = len(self.normal_indices) // batch_size
    
    def __iter__(self):
        # Shuffle both class indices
        normal = self.normal_indices.copy()
        seizure = self.seizure_indices.copy()
        random.shuffle(normal)
        random.shuffle(seizure)
        
        if len(seizure) == 0:
            # Fallback: only normal samples (shouldn't happen in training)
            for i in range(0, len(normal) - self.batch_size + 1, self.batch_size):
                yield from normal[i:i+self.batch_size]
            return
        
        # Extend normal samples if needed (with replacement)
        n_normal_needed = self.n_batches * self.normal_per_batch
        if len(normal) < n_normal_needed:
            normal = (normal * (n_normal_needed // len(normal) + 1))[:n_normal_needed]
        else:
            normal = normal[:n_normal_needed]
        
        # Extend seizure samples with repetition if needed
        n_seizure_needed = self.n_batches * self.seizure_per_batch
        if len(seizure) < n_seizure_needed:
            seizure = (seizure * (n_seizure_needed // len(seizure) + 1))[:n_seizure_needed]
        
        # Create balanced batches
        for i in range(self.n_batches):
            batch = []
            batch.extend(seizure[i*self.seizure_per_batch:(i+1)*self.seizure_per_batch])
            batch.extend(normal[i*self.normal_per_batch:(i+1)*self.normal_per_batch])
            random.shuffle(batch)  # Shuffle within batch to avoid ordering bias
            yield from batch
    
    def __len__(self):
        return self.n_batches * self.batch_size

class FastEEGDatasetV2(Dataset):
    """
    Optimized Dataset with Numpy Memmap (Disk-backed).
    Allows loading Terabytes of data with < 1GB RAM usage.
    """
    def __init__(self, data_dir: str, dataset_name: str = 'chbmit', 
                 subjects: List[str] = None, preictal_window: int = 15,
                 n_channels: int = 19, max_epochs: int = 50000, 
                 augment: bool = False, binary_modes: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.augment = augment
        self.binary = binary_modes
        
        # 1. Identify Subjects
        self.cutoff_window = preictal_window
        pattern = "*.npz"
        data_path = Path(data_dir) / dataset_name
        
        # Handle case where data_dir doesn't end in 'dataset'
        if not data_path.exists():
            pass 

        # [v5.1 FIX] Explicit Subject Selection
        # Previously 'exclude_subjects' caused train/val split inversion.
        # Now we strictly include only the requested subjects.
        all_files = sorted([f for f in data_path.rglob(pattern) if "processed" not in str(f)])
        
        if subjects is not None and len(subjects) > 0:
            self.subjects = [
                f for f in all_files 
                if any(s in f.name for s in subjects)  # Check if subject ID is in filename
            ]
        else:
            print("[WARN] No subject list provided, loading ALL subjects found.")
            self.subjects = all_files
        
        if not self.subjects:
            print(f"[WARN] No matched subjects found in {data_path}")
            self.file_maps = []
            self.cumulative_sizes = [0]
            self.class_weights = torch.ones(2)
            return

        # 2. Serial Pre-Processing
        print(f"[FastEEGDatasetV2] Verifying memmap caches for {len(self.subjects)} subjects...")
        self._ensure_all_caches(self.subjects, preictal_window, max_epochs)
        
        # 3. Fast Loading (Memmap)
        print(f"[FastEEGDatasetV2] Initializing memory maps...")
        
        self.file_maps = [] # List of dicts: {'X': mmap, 'y': array, 'ctx': mmap}
        self.y_list = [] # Keep labels in memory for weighting
        
        processed_dir = Path("cache") / "processed_mmap"
        if not processed_dir.exists():
            processed_dir.mkdir(parents=True, exist_ok=True)

        total_epochs = 0
        
        for subj_path in self.subjects:
            base_name = f"{subj_path.stem}_w{preictal_window}_ch{self.n_channels}_v5"
            f_X = processed_dir / f"{base_name}_X.npy"
            f_y = processed_dir / f"{base_name}_y.npy"
            f_ctx = processed_dir / f"{base_name}_ctx.npy"
            
            if f_X.exists() and f_y.exists():
                try:
                    # Load labels into RAM (small)
                    y_data = np.load(f_y)
                    
                    # Memmap features (Zero RAM)
                    # We open in 'r' mode which is read-only shared
                    X_mmap = np.load(f_X, mmap_mode='r')
                    ctx_mmap = np.load(f_ctx, mmap_mode='r')
                    
                    self.file_maps.append({
                        'X': X_mmap,
                        'y': y_data,
                        'ctx': ctx_mmap
                    })
                    
                    self.y_list.append(torch.from_numpy(y_data))
                    total_epochs += len(y_data)
                    
                except Exception as e:
                    print(f"[ERROR] Could not load memmap {base_name}: {e}")
            else:
                print(f"[WARN] Cache missing for {subj_path.stem}")

        # Indexing logic
        self.cumulative_sizes = np.cumsum([0] + [len(m['y']) for m in self.file_maps])
        
        if total_epochs > 0:
            # Calculate Class Weights
            full_y = torch.cat(self.y_list, dim=0)
            if self.binary:
                 full_y = (full_y > 0).long()
                 
            classes, counts = torch.unique(full_y, return_counts=True)
            self.class_dist = {int(c): int(ct) for c, ct in zip(classes, counts)}
            
            count_list = [self.class_dist.get(i, 0) for i in range(len(classes))]
            total = sum(count_list)
            # Standard inverse frequency
            weights = [total / (len(classes) * max(1, c)) for c in count_list]
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

            print(f"  Total epochs: {total_epochs}")
            print(f"  Mode: Memmap (Features on disk)")
            print(f"  Class distribution: {self.class_dist}")
        else:
            self.class_weights = torch.ones(2)

    def _ensure_all_caches(self, subjects, preictal_window, max_epochs):
        processed_dir = Path("cache") / "processed_mmap"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        for i, path in enumerate(subjects):
            base_name = f"{path.stem}_w{preictal_window}_ch{self.n_channels}_v5"
            f_X = processed_dir / f"{base_name}_X.npy"
            
            if not f_X.exists():
                print(f"  [PROCESS] Generating memmap for {path.stem} ({i+1}/{len(subjects)})...")
                self._process_and_save_memmap(path, processed_dir, base_name, preictal_window, max_epochs)
                gc.collect()

    def _process_and_save_memmap(self, path, processed_dir, base_name, preictal_window, max_epochs):
        """
        Process single subject and stream-save as .npy files (Safe for 8GB RAM).
        """
        try:
            with np.load(path, mmap_mode='r') as data:
                if 'y' not in data or 'X' not in data: return
                
                # 1. Determine Selection
                y_orig = data['y'][:]
                y = self._create_3class_labels(y_orig, preictal_window)
                
                # Keep everything unless requested otherwise. 
                # Memmap allows "infinite" data so we just keep valid indices.
                keep_idx = np.arange(len(y))
                total_samples = len(keep_idx)
                
                # 2. Setup Dimensions
                X_full_mmap = data['X'] 
                n_time = X_full_mmap.shape[2]
                out_channels = self.n_channels
                
                # 3. Create Output Memmaps (Stream Writing)
                f_X = processed_dir / f"{base_name}_X.npy"
                f_y = processed_dir / f"{base_name}_y.npy"
                f_ctx = processed_dir / f"{base_name}_ctx.npy"
                
                # Context dim: channels * 4 features * 2 (duplicated)
                ctx_dim = out_channels * 8 
                
                # Create .npy directly on disk
                X_out = np.lib.format.open_memmap(
                    f_X, mode='w+', dtype=np.float16, 
                    shape=(total_samples, out_channels, n_time)
                )
                
                ctx_out = np.lib.format.open_memmap(
                    f_ctx, mode='w+', dtype=np.float16,
                    shape=(total_samples, ctx_dim)
                )
                
                # y is small, just save at end
                y_final = y[keep_idx]
                
                # 4. Stream Process
                chunk_size = 500
                
                for i in range(0, total_samples, chunk_size):
                    chunk_indices = keep_idx[i:i+chunk_size]
                    current_chunk_size = len(chunk_indices)
                    
                    # Load Chunk (RAM usage: ~500 epochs * 19ch * 1280 floats * 4B = ~48MB)
                    X_chunk = X_full_mmap[chunk_indices].astype(np.float32)
                    
                    # Standardize Channels
                    if X_chunk.shape[1] > out_channels:
                        X_chunk = X_chunk[:, :out_channels, :]
                    elif X_chunk.shape[1] < out_channels:
                        padding = np.zeros((current_chunk_size, out_channels - X_chunk.shape[1], n_time), dtype=np.float32)
                        X_chunk = np.concatenate([X_chunk, padding], axis=1)
                    
                    # Robust Scale
                    y_chunk = y[chunk_indices]
                    X_chunk = self._robust_scale(X_chunk, y_chunk)
                    
                    # Context
                    ctx_chunk = self._extract_fast_context(X_chunk)
                    
                    # Write to disk
                    X_out[i:i+current_chunk_size] = X_chunk.astype(np.float16)
                    ctx_out[i:i+current_chunk_size] = ctx_chunk.astype(np.float16)
                    
                    # Explicit flush every now and then (though OS does it)
                    if i % 5000 == 0:
                        X_out.flush()
                        ctx_out.flush()
                        gc.collect()

                # Finalize
                X_out.flush()
                ctx_out.flush()
                np.save(f_y, y_final)
                
                # Cleanup file handles
                del X_out
                del ctx_out
                gc.collect()
                
        except Exception as e:
            print(f"[ERROR] Failed processing {path.name}: {e}")
            # If failed, try to delete corrupted files
            try:
                if f_X.exists(): f_X.unlink()
                if f_ctx.exists(): f_ctx.unlink()
            except: pass

    def _robust_scale(self, X, y):
        # [v5.1 FIX] Fixed Constant Scaling
        # Instead of normalizing each window to unit variance (which destroys
        # amplitude information critical for seizure detection), we scale by
        # a fixed constant (approximate interquartile range of normal EEG).
        # This preserves the relative amplitude of seizures vs normal.
        
        # FIXED_SCALE determined empirically from CHB-MIT background
        # Normal EEG is approx +/- 20-50 uV. Seizures +/- 500 uV.
        # Dividing by 20.0 keeps normal ~1.0, seizures ~10.0+
        FIXED_IQR = 20.0 
        
        # Center by median (remove DC offset)
        med = np.median(X, axis=1, keepdims=True)
        
        return ((X - med) / FIXED_IQR).astype(np.float32)
    
    def _create_3class_labels(self, y_orig: np.ndarray, preictal_window: int) -> np.ndarray:
        """Convert to 3-class: 0=interictal, 1=preictal, 2=ictal."""
        y = np.zeros(len(y_orig), dtype=np.int64)
        ictal_mask = y_orig > 0
        y[ictal_mask] = 2
        
        ictal_indices = np.where(ictal_mask)[0]
        if len(ictal_indices) == 0: return y
        
        diffs = np.diff(ictal_indices)
        starts = [ictal_indices[0]] + [ictal_indices[i+1] for i in range(len(diffs)) if diffs[i] > 1]
        
        for start in starts:
            s_idx = max(0, start - preictal_window)
            e_idx = start
            for i in range(s_idx, e_idx):
                if y[i] == 0:
                    y[i] = 1
        return y

    def _extract_fast_context(self, X: np.ndarray, fs: int = 256) -> np.ndarray:
        """
        Extract spectral band powers + time-domain features for FiLM conditioning.
        
        v4.5 IMPROVED: Replaced line_length with kurtosis (better spike detection)
        
        Computes 8 unique features per channel:
        - 5 spectral bands: delta (0.5-4Hz), theta (4-8Hz), alpha (8-13Hz), 
                           beta (13-30Hz), gamma (30-70Hz)
        - 3 time-domain: log energy, kurtosis (spike detection), peak-to-peak amplitude
        
        Total: 19 channels × 8 features = 152 dimensions
        """
        B, C, T = X.shape
        
        # === Spectral Band Powers (FFT-based) ===
        # Band definitions in Hz - clinically relevant frequency bands
        bands = [
            (0.5, 4),   # Delta - deep sleep, pathology indicator
            (4, 8),     # Theta - drowsiness, memory
            (8, 13),    # Alpha - relaxation, eyes closed
            (13, 30),   # Beta - active thinking, alertness
            (30, 70),   # Gamma - high cognitive function, seizure activity
        ]
        
        # Compute FFT power spectrum
        freqs = np.fft.rfftfreq(T, 1.0 / fs)
        fft_result = np.fft.rfft(X, axis=2)
        power_spectrum = np.abs(fft_result) ** 2
        
        # Extract band powers
        band_powers = []
        for low, high in bands:
            # Create frequency mask for this band
            mask = (freqs >= low) & (freqs < high)
            if mask.sum() > 0:
                # Mean power in band (log scale for stability)
                band_power = np.log(power_spectrum[:, :, mask].mean(axis=2) + 1e-8)
            else:
                # Fallback if no frequencies in band
                band_power = np.zeros((B, C), dtype=np.float32)
            band_powers.append(band_power)
        
        # Stack spectral features: (B, C, 5)
        spectral_feats = np.stack(band_powers, axis=2)
        
        # === Time-Domain Features ===
        # Log energy (variance)
        energy = np.log(np.var(X, axis=2, keepdims=True) + 1e-8)
        
        # Kurtosis (REPLACES line_length - better for detecting seizure spikes)
        # Kurtosis measures "tailedness" - high values indicate sharp spikes
        # Formula: E[(X-μ)⁴] / σ⁴ - 3 (excess kurtosis)
        mean_x = np.mean(X, axis=2, keepdims=True)
        std_x = np.std(X, axis=2, keepdims=True) + 1e-8
        x_centered = (X - mean_x) / std_x
        kurtosis = np.mean(x_centered ** 4, axis=2, keepdims=True) - 3.0  # Excess kurtosis
        kurtosis = np.clip(kurtosis, -10, 50)  # Clamp for stability
        
        # Peak-to-peak amplitude
        amplitude = (np.max(X, axis=2, keepdims=True) - np.min(X, axis=2, keepdims=True))
        
        # Stack time-domain features: (B, C, 3)
        time_feats = np.concatenate([energy, kurtosis, amplitude], axis=2)
        
        # === Combine All Features ===
        # Total: (B, C, 8) = 5 spectral + 3 time-domain
        all_feats = np.concatenate([spectral_feats, time_feats], axis=2)
        
        # Flatten to (B, C * 8) = (B, 152) for 19 channels
        context = all_feats.reshape(B, -1)
        
        return context.astype(np.float32)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Lookup Logic
        subj_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1
        
        if subj_idx < 0 or subj_idx >= len(self.file_maps):
            return torch.zeros(1), torch.zeros(1), torch.tensor(0)

        local_idx = idx - self.cumulative_sizes[subj_idx]
        
        try:
            # Memmap access (reads from disk)
            data_dict = self.file_maps[subj_idx]
            
            # Copy to RAM as float32 for PyTorch
            # This is the only moment we use RAM for this sample
            x = torch.from_numpy(data_dict['X'][local_idx].astype(np.float32))
            y = torch.tensor(data_dict['y'][local_idx], dtype=torch.long)
            ctx = torch.from_numpy(data_dict['ctx'][local_idx].astype(np.float32))
            
        except Exception:
             return torch.zeros(self.n_channels, 1280), torch.zeros(152), torch.tensor(0)
        
        # Binary mapping (do this before augmentation to know label)
        if self.binary and y == 2:
            y = torch.tensor(1, dtype=torch.long)
        
        # Augmentations (On the fly)
        if self.augment and x.shape[0] > 1:
            # TEMPORAL-AWARE AUGMENTATION for seizure samples
            # Seizures get stronger augmentation to increase effective sample count
            is_seizure = (y.item() == 1) if isinstance(y, torch.Tensor) else (y == 1)
            
            # Time-shift augmentation (stronger for seizure)
            shift_prob = 0.5 if is_seizure else 0.2
            shift_range = 50 if is_seizure else 20  # Larger shifts for seizure
            if np.random.random() < shift_prob:
                shift = np.random.randint(-shift_range, shift_range)
                x = torch.roll(x, shift, dims=-1)
            
            # Gaussian noise (stronger for seizure)
            noise_prob = 0.35 if is_seizure else 0.15
            noise_scale = 0.04 if is_seizure else 0.02
            if np.random.random() < noise_prob:
                noise_level = noise_scale * x.std()
                x = x + torch.randn_like(x) * noise_level
            
            # Amplitude scaling
            scale_prob = 0.30 if is_seizure else 0.15
            if np.random.random() < scale_prob:
                scale = np.random.uniform(0.85, 1.15)
                x = x * scale
            
            # Time-stretch for seizure samples (subtle jitter)
            if is_seizure and np.random.random() < 0.2:
                # Random segment reversal (preserves spectral content)
                T = x.shape[-1]
                seg_start = np.random.randint(0, T // 2)
                seg_end = seg_start + T // 4
                x[:, seg_start:seg_end] = x[:, seg_start:seg_end].flip(dims=[-1])

        return x, ctx, y
    
    def get_sample_weights(self) -> torch.Tensor:
        if self.__len__() == 0: return torch.zeros(0)
        
        # Construct full weights vector
        weights_list = []
        for y_tensor in self.y_list:
            # Map labels to weights
            y_mapped = y_tensor.clone()
            if self.binary:
                y_mapped[y_mapped == 2] = 1
            
            w_chunk = self.class_weights[y_mapped]
            weights_list.append(w_chunk)
            
        return torch.cat(weights_list)


def create_optimized_loaders(
    cache_dir: str,
    dataset: str,
    train_subjects: List[str],
    val_subjects: List[str],
    n_channels: int = 19,
    batch_size: int = 64,
    val_batch_size: int = None,  # New param for efficiency
    num_workers: int = 0, # FORCE 0 for Windows/Memmap safety
    max_epochs_per_subject: int = 100000,
    binary_modes: bool = True
) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    
    print(f"\n[DataLoaders] Creating BALANCED loaders (v4.4)...")
    
    # Train Dataset
    train_dataset = FastEEGDatasetV2(
        cache_dir, dataset, subjects=train_subjects, # [v5.1 FIX] Use explicit inclusion
        n_channels=n_channels,
        augment=True,
        max_epochs=max_epochs_per_subject,
        binary_modes=binary_modes
    )
    
    # Val Dataset
    val_dataset = FastEEGDatasetV2(
        cache_dir, dataset, subjects=val_subjects, # [v5.1 FIX] Use explicit inclusion
        n_channels=n_channels,
        augment=False,
        max_epochs=max_epochs_per_subject, 
        binary_modes=binary_modes
    )

    # v4.4 CRITICAL FIX: Use BalancedBatchSampler instead of WeightedRandomSampler
    # This avoids the double-penalty that caused F1-Seizure stagnation
    sampler = None
    if len(train_dataset) > 0 and len(train_dataset.y_list) > 0:
        # Get all labels for balanced sampling
        all_labels = torch.cat(train_dataset.y_list, dim=0)
        if binary_modes:
            all_labels = (all_labels > 0).long()
        
        # Create balanced batch sampler: 40% seizure, 60% normal per batch
        sampler = BalancedBatchSampler(
            all_labels, 
            batch_size=batch_size,
            minority_ratio=0.4,
            drop_last=True
        )
        print(f"  [BalancedBatchSampler] {sampler.seizure_per_batch} seizure + "
              f"{sampler.normal_per_batch} normal per batch, {sampler.n_batches} batches/epoch")
    
    # Force num_workers=0 to avoid pickling memmaps across processes on Windows
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False, drop_last=True
    )
    
    if val_batch_size is None:
        val_batch_size = batch_size

    val_loader = DataLoader(
        val_dataset, batch_size=val_batch_size, shuffle=False,
        num_workers=0, pin_memory=False
    )
    
    return train_loader, val_loader, train_dataset.class_weights

def get_subject_list(cache_dir: str, dataset: str) -> List[str]:
    """Get cached subject IDs."""
    path = Path(cache_dir) / dataset
    pattern = "*.npz"
    subjects = sorted([f.stem.replace('_raw', '') for f in path.rglob(pattern) if "processed" not in str(f)])
    return subjects

def get_min_channels(cache_dir: str, dataset: str, subjects: List[str]) -> int:
    """Get minimum channel count."""
    min_ch = 999
    path = Path(cache_dir) / dataset
    
    count = 0
    for f in path.rglob("*.npz"):
        if "processed" in str(f): continue
        if count > 5: break
        try:
            with np.load(f) as data:
                if 'X' in data:
                    min_ch = min(min_ch, data['X'].shape[1])
                    count += 1
        except:
            pass
            
    return min_ch if min_ch < 999 else 19

