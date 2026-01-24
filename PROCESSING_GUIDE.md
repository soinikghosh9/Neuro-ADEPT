# High-Performance Data Processing Guide

This guide explains how to use your 64GB RAM workstation to rapidly preprocess the CHB-MIT and SIENA datasets and generate the cache files required for training Neuro-ADEPT.

## 1. Setup on High-RAM Machine

Ensure the machine has:
*   Python 3.10+
*   The project code (clone this repo)
*   Raw datasets downloaded:
    *   `datasets/chbmit/*.edf`
    *   `datasets/siena/*.edf`

Install dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib mne torch tqdm
```

## 2. Step 1: Convert EDF to Raw NPZ (Parallelized)

We have optimized `preprocess.py` to use multiple cores. Run this command to convert raw EDFs to `.npz` format.

```bash
# Process both datasets with maximum parallelism
# --workers: defaults to CPU count. Set manually if needed (e.g., --workers 16)
python preprocess.py --all --workers 16
```

**Output**: This will create:
*   `cache/chbmit/*.npz`
*   `cache/siena/*.npz`

## 3. Step 2: Generate Training Cache (Memmaps)

This step normally happens during the first training run, which can be slow. Run it explicitly on the fast machine to generate the optimization-ready files.

```bash
python generate_cache.py --dataset chbmit --n_workers 8
python generate_cache.py --dataset siena --n_workers 8
```

**Output**: This will create:
*   `cache/processed_mmap/*.npy` (Features, Labels, Context)

## 4. Transfer to Training Machine

Once processing is complete, transfer the `cache` directory to your training machine (the one with the RTX 5070).

1.  Zip the cache:
    ```bash
    # Windows PowerShell
    Compress-Archive -Path cache -DestinationPath neuro_cache.zip
    ```
2.  Move `neuro_cache.zip` to the training machine.
3.  Unzip it in the project root:
    ```bash
    Expand-Archive -Path neuro_cache.zip -DestinationPath . -Force
    ```

## 5. Summary of Files

| File Type | Location | Purpose |
| :--- | :--- | :--- |
| **Raw EDF** | `datasets/` | Original data (keep on 64GB machine) |
| **Raw NPZ** | `cache/{dataset}/` | Intermediate epoch format |
| **Memmaps** | `cache/processed_mmap/` | **Ready for Training** (Fast I/O) |

You are now ready to train with maximum efficiency!
