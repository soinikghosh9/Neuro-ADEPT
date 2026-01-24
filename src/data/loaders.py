"""
Optimized EEG Data Loaders for CHB-MIT and SIENA Datasets.

Based on GenEEG (https://github.com/soinikghosh9/GenEEG) parsing patterns.

This module provides efficient data loading with:
- Robust seizure annotation parsing from summary/list files
- 3-class labeling: Normal (0), Preictal (1), Ictal (2)
- Memory-efficient lazy loading
- NERVE-ML compliant data handling

Supports:
- CHB-MIT: 23 channels, 256 Hz, US line frequency (60 Hz)
- SIENA: 29+ channels, 512 Hz, EU line frequency (50 Hz)
"""

import re
import os
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterator, Any
from abc import ABC, abstractmethod

# Suppress MNE warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="mne")
warnings.filterwarnings("ignore", message=".*highpass.*")
warnings.filterwarnings("ignore", message=".*lowpass.*")

try:
    import mne
    mne.set_log_level('ERROR')
    from mne.io import read_raw_edf
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    print("Warning: MNE not installed. Install with: pip install mne")


# ==============================================================================
# Utility Functions
# ==============================================================================

def time_to_seconds(time_str: str) -> int:
    """
    Convert time string to seconds.
    
    Args:
        time_str: Time string in format 'HH:MM:SS' or 'HH.MM.SS'
    
    Returns:
        Time in seconds
    """
    # Handle both ':' and '.' as separators
    time_str = time_str.strip().replace('.', ':')
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def normalize_filename(filename: str) -> str:
    """
    Normalize filename for consistent matching.
    
    Handles common issues:
    - Case insensitivity
    - O vs 0 (letter vs digit) substitution (common OCR/typo issue)
    """
    normalized = filename.lower().strip()
    # Replace letter 'o' with digit '0' after 'pn' prefix for SIENA files
    # This handles PNO6 -> PN06 typo issue
    if normalized.startswith('pn'):
        # Replace 'o' with '0' in the subject ID portion (after 'pn')
        prefix = 'pn'
        rest = normalized[2:]
        # Replace letter 'o' with digit '0' in subject number
        rest = rest.replace('o', '0')
        normalized = prefix + rest
    return normalized


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class SeizureAnnotation:
    """Single seizure event annotation."""
    onset: float           # Onset time in seconds from recording start
    offset: float          # Offset time in seconds
    duration: float = 0.0  # Duration in seconds
    
    def __post_init__(self):
        self.duration = self.offset - self.onset


@dataclass
class Recording:
    """Single EEG recording with metadata."""
    file_path: str
    file_name: str
    subject_id: str
    seizures: List[SeizureAnnotation] = field(default_factory=list)
    duration: float = 0.0     # Recording duration in seconds
    n_channels: int = 0
    fs: float = 256.0         # Sampling frequency
    data: Optional[np.ndarray] = None  # Lazy loaded
    channel_names: List[str] = field(default_factory=list)
    
    @property
    def has_seizure(self) -> bool:
        return len(self.seizures) > 0


@dataclass
class SubjectData:
    """All data for a single subject."""
    subject_id: str
    recordings: List[Recording] = field(default_factory=list)
    
    @property
    def n_seizures(self) -> int:
        return sum(len(r.seizures) for r in self.recordings)
    
    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.recordings)


# ==============================================================================
# CHB-MIT Loader (GenEEG Pattern)
# ==============================================================================

class CHBMITLoader:
    """
    CHB-MIT Scalp EEG Database Loader.
    
    Based on GenEEG parsing patterns.
    
    Dataset: https://physionet.org/content/chbmit/1.0.0/
    
    - 22 pediatric subjects with epilepsy
    - 198 seizures across all subjects
    - 256 Hz sampling rate
    - 23 EEG channels (10-20 system, bipolar montage)
    
    File Structure Expected:
        chbmit/
        ├── chb01/
        │   ├── chb01-summary.txt  (seizure annotations)
        │   ├── chb01_01.edf
        │   └── ...
        └── chb02/
            └── ...
    """
    
    FS = 256.0  # Hz
    
    def __init__(
        self,
        root_dir: str,
        target_sampling_rate: int = 256,
        preictal_duration_sec: int = 1200,  # 20 minutes
        bandpass_low: float = 0.5,
        bandpass_high: float = 70.0,
        notch_freq: float = 60.0,  # US standard
        verbose: bool = True,
        load_data: bool = False  # Lazy loading by default
    ):
        """Initialize CHB-MIT loader.
        
        Args:
            load_data: If False (default), only metadata is loaded. Data is loaded
                      on-demand via load_recording_data(). This is memory-efficient.
        """
        self.root_dir = Path(root_dir)
        self.target_sr = target_sampling_rate
        self.preictal_duration = preictal_duration_sec
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.verbose = verbose
        self.fs = self.FS
        self.load_data = load_data
        
        self.subjects: Dict[str, SubjectData] = {}
        self._loaded = False
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"CHB-MIT dataset not found at {root_dir}")
    
    def load_all(self, subjects: Optional[List[str]] = None, load_data: Optional[bool] = None) -> None:
        """Load all subjects from directory.
        
        Args:
            subjects: List of subject IDs to load. If None, loads all.
            load_data: Override instance load_data setting for this call.
        """
        if load_data is not None:
            self.load_data = load_data
            
        if self.verbose:
            mode = "with data" if self.load_data else "metadata only (lazy)"
            print(f"Loading CHB-MIT dataset from: {self.root_dir} [{mode}]")
        
        # Discover patient directories
        patient_dirs = self._discover_patients()
        
        if subjects:
            patient_dirs = [d for d in patient_dirs if d.name in subjects]
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            try:
                self._load_patient(patient_dir, patient_id)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load {patient_id}: {e}")
        
        self._loaded = True
        print(f"Loaded {len(self.subjects)} subjects")
    
    def _discover_patients(self) -> List[Path]:
        """Discover all patient directories."""
        patient_dirs = []
        
        for item in sorted(self.root_dir.iterdir()):
            if item.is_dir() and item.name.startswith('chb'):
                # Check for summary file
                summary_file = item / f"{item.name}-summary.txt"
                if summary_file.exists():
                    patient_dirs.append(item)
                    if self.verbose:
                        print(f"  Found patient: {item.name}")
                else:
                    # Check for summary in parent or any summary file
                    summary_files = list(item.glob("*-summary.txt"))
                    if summary_files:
                        patient_dirs.append(item)
                        if self.verbose:
                            print(f"  Found patient: {item.name}")
        
        return patient_dirs
    
    def _load_patient(self, patient_dir: Path, patient_id: str) -> None:
        """Load a single patient."""
        # Parse seizure annotations using GenEEG pattern
        annotations = self.parse_summary_file(patient_id, patient_dir)
        
        # Find all EDF files
        edf_files = sorted(patient_dir.glob("*.edf"))
        
        if not edf_files:
            if self.verbose:
                print(f"  No EDF files found for {patient_id}")
            return
        
        subject_data = SubjectData(subject_id=patient_id)
        
        for edf_path in edf_files:
            file_name = edf_path.name
            
            # Get seizures for this file
            seizure_list = annotations.get(file_name, [])
            seizures = [SeizureAnnotation(onset=s['start'], offset=s['end']) 
                       for s in seizure_list]
            
            if self.load_data:
                # Full loading - load data immediately
                try:
                    data, fs, channel_names = self._load_edf_file(str(edf_path))
                    
                    recording = Recording(
                        file_path=str(edf_path),
                        file_name=file_name,
                        subject_id=patient_id,
                        seizures=seizures,
                        duration=data.shape[1] / fs,
                        n_channels=data.shape[0],
                        fs=fs,
                        data=data,
                        channel_names=channel_names
                    )
                    subject_data.recordings.append(recording)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: Failed to load {file_name}: {e}")
            else:
                # Lazy loading - store metadata only, get duration from EDF header
                try:
                    duration, n_channels, fs, channel_names = self._get_edf_metadata(str(edf_path))
                    
                    recording = Recording(
                        file_path=str(edf_path),
                        file_name=file_name,
                        subject_id=patient_id,
                        seizures=seizures,
                        duration=duration,
                        n_channels=n_channels,
                        fs=fs,
                        data=None,  # Lazy - loaded on demand
                        channel_names=channel_names
                    )
                    subject_data.recordings.append(recording)
                    
                except Exception as e:
                    if self.verbose:
                        print(f"    Warning: Failed to get metadata for {file_name}: {e}")
        
        if subject_data.recordings:
            self.subjects[patient_id] = subject_data
            if self.verbose:
                print(f"  {patient_id}: {len(subject_data.recordings)} recordings, "
                      f"{subject_data.n_seizures} seizures")
    
    def _get_edf_metadata(self, file_path: str) -> Tuple[float, int, float, List[str]]:
        """Get EDF file metadata without loading data (memory efficient)."""
        if not HAS_MNE:
            raise ImportError("MNE is required. Install with: pip install mne")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # preload=False means data is not loaded into memory
            raw = read_raw_edf(file_path, preload=False, verbose=False)
        
        duration = raw.n_times / raw.info['sfreq']
        n_channels = len(raw.ch_names)
        fs = raw.info['sfreq']
        channel_names = raw.ch_names
        
        # Close the file handle
        del raw
        
        return duration, n_channels, fs, channel_names
    
    def load_recording_data(self, recording: Recording) -> np.ndarray:
        """Load data for a specific recording on-demand.
        
        Args:
            recording: Recording object to load data for
            
        Returns:
            EEG data array (n_channels, n_samples)
        """
        if recording.data is not None:
            return recording.data
        
        data, fs, channel_names = self._load_edf_file(recording.file_path)
        recording.data = data
        recording.fs = fs
        recording.channel_names = channel_names
        recording.n_channels = data.shape[0]
        recording.duration = data.shape[1] / fs
        
        return data
    
    def parse_summary_file(self, patient_id: str, patient_dir: Path) -> Dict[str, List[Dict]]:
        """
        Parse patient summary file to extract seizure annotations.
        
        GenEEG pattern: handles both standard and alternative formats.
        
        Returns:
            Dictionary mapping filename to list of seizure events:
            {'chb01_03.edf': [{'start': 2996, 'end': 3036}, ...]}
        """
        # Find summary file
        summary_path = patient_dir / f"{patient_id}-summary.txt"
        
        if not summary_path.exists():
            # Try to find any summary file
            summary_files = list(patient_dir.glob("*-summary.txt"))
            if summary_files:
                summary_path = summary_files[0]
            else:
                if self.verbose:
                    print(f"  Warning: Summary file not found for {patient_id}")
                return {}
        
        if self.verbose:
            print(f"  Reading summary file: {summary_path.name}")
        
        annotations = {}
        current_file = None
        
        with open(summary_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Detect file entry
            if line.startswith('File Name:'):
                current_file = line.split(':', 1)[1].strip()
                annotations[current_file] = []
                i += 1
                continue
            
            # Detect number of seizures
            if line.startswith('Number of Seizures in File:'):
                num_seizures_str = line.split(':', 1)[1].strip()
                num_seizures = int(num_seizures_str)
                
                # Parse each seizure in this file
                for _ in range(num_seizures):
                    i += 1
                    # Seizure Start Time
                    if i < len(lines) and 'Seizure' in lines[i] and 'Start' in lines[i]:
                        start_line = lines[i].strip()
                        # Match number before "seconds"
                        match = re.search(r'(\d+)\s*seconds?', start_line)
                        if match:
                            start_sec = int(match.group(1))
                        else:
                            # Try alternative format without "seconds"
                            try:
                                start_sec = int(start_line.split(':', 1)[1].strip().split()[0])
                            except:
                                continue
                        
                        i += 1
                        # Seizure End Time
                        if i < len(lines) and 'Seizure' in lines[i] and 'End' in lines[i]:
                            end_line = lines[i].strip()
                            match = re.search(r'(\d+)\s*seconds?', end_line)
                            if match:
                                end_sec = int(match.group(1))
                            else:
                                try:
                                    end_sec = int(end_line.split(':', 1)[1].strip().split()[0])
                                except:
                                    continue
                            
                            if current_file:
                                annotations[current_file].append({
                                    'start': start_sec,
                                    'end': end_sec
                                })
            
            i += 1
        
        if self.verbose and annotations:
            total_seizures = sum(len(v) for v in annotations.values())
            files_with_seizures = sum(1 for v in annotations.values() if len(v) > 0)
            print(f"    Found {files_with_seizures} files with {total_seizures} total seizures")
        
        return annotations
    
    def _load_edf_file(self, file_path: str, preprocess: bool = True) -> Tuple[np.ndarray, float, List[str]]:
        """Load and preprocess a single EDF file."""
        if not HAS_MNE:
            raise ImportError("MNE is required. Install with: pip install mne")
        
        # Load with MNE
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = read_raw_edf(file_path, preload=True, verbose=False)
        
        if preprocess:
            # Apply bandpass filter
            raw.filter(
                l_freq=self.bandpass_low,
                h_freq=self.bandpass_high,
                method='iir',
                iir_params={'order': 4, 'ftype': 'butter'},
                verbose=False
            )
            
            # Apply notch filter for power line noise
            raw.notch_filter(
                freqs=self.notch_freq,
                method='iir',
                verbose=False
            )
            
            # Resample if needed
            original_sr = int(raw.info['sfreq'])
            if original_sr != self.target_sr:
                raw.resample(self.target_sr, npad='auto', verbose=False)
            
            # Common average reference
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except:
                pass  # Skip if reference can't be set
        
        # Extract data
        data = raw.get_data()  # (n_channels, n_samples)
        channel_names = raw.ch_names
        sampling_rate = int(raw.info['sfreq'])
        
        return data, sampling_rate, channel_names
    
    def get_subject_ids(self) -> List[str]:
        """Get list of loaded subject IDs."""
        return list(self.subjects.keys())
    
    def get_subject_data(self, subject_id: str) -> SubjectData:
        """Get data for a specific subject."""
        if subject_id not in self.subjects:
            raise KeyError(f"Subject {subject_id} not found. Available: {self.get_subject_ids()}")
        return self.subjects[subject_id]


# ==============================================================================
# SIENA Loader (GenEEG Pattern)
# ==============================================================================

class SIENALoader:
    """
    SIENA Scalp EEG Database Loader.
    
    Based on GenEEG parsing patterns.
    
    Dataset: https://physionet.org/content/siena-scalp-eeg/1.0.0/
    
    - 14 subjects with epilepsy
    - 47 seizures across all subjects
    - 512 Hz sampling rate
    - 29+ channels
    
    Summary file format (Siena v3.0.0):
        File name: PN00-1.edf
        Registration start time: 11:42:54
        Seizure start time: 12:15:30
        Seizure end time: 12:16:15
    """
    
    FS = 512.0  # Hz
    
    def __init__(
        self,
        root_dir: str,
        original_sampling_rate: int = 512,
        target_sampling_rate: int = 256,
        num_channels: int = 16,
        preictal_duration_sec: int = 1200,  # 20 minutes
        bandpass_low: float = 0.5,
        bandpass_high: float = 70.0,
        notch_freq: float = 50.0,  # European standard
        verbose: bool = True,
        load_data: bool = False  # Lazy loading by default
    ):
        """Initialize SIENA loader.
        
        Args:
            load_data: If False (default), only metadata is loaded. Data is loaded
                      on-demand via load_recording_data(). This is memory-efficient.
        """
        self.root_dir = Path(root_dir)
        self.original_sr = original_sampling_rate
        self.target_sr = target_sampling_rate
        self.num_channels = num_channels
        self.preictal_duration = preictal_duration_sec
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.verbose = verbose
        self.fs = self.FS
        self.load_data = load_data
        
        self.subjects: Dict[str, SubjectData] = {}
        self._loaded = False
        
        # Preferred channel order (10-20 system)
        self.preferred_channels = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
            'Fz', 'Cz', 'Pz'
        ]
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"SIENA dataset not found at {root_dir}")
    
    def load_all(self, subjects: Optional[List[str]] = None, load_data: Optional[bool] = None) -> None:
        """Load all subjects from directory.
        
        Args:
            subjects: List of subject IDs to load. If None, loads all.
            load_data: Override instance load_data setting for this call.
        """
        if load_data is not None:
            self.load_data = load_data
            
        if self.verbose:
            mode = "with data" if self.load_data else "metadata only (lazy)"
            print(f"Loading SIENA dataset from: {self.root_dir} [{mode}]")
        
        # Discover patient directories
        patient_dirs = []
        for item in sorted(self.root_dir.iterdir()):
            if item.is_dir() and item.name.upper().startswith('PN'):
                patient_dirs.append(item)
        
        if subjects:
            patient_dirs = [d for d in patient_dirs if d.name in subjects]
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            try:
                self._load_patient(patient_dir, patient_id)
            except Exception as e:
                if self.verbose:
                    print(f"  Warning: Failed to load {patient_id}: {e}")
        
        self._loaded = True
        print(f"Loaded {len(self.subjects)} subjects")
    
    def _load_patient(self, patient_dir: Path, patient_id: str) -> None:
        """Load a single patient."""
        # Parse seizure annotations using GenEEG pattern
        annotations = self.parse_summary_file(patient_id, patient_dir)
        
        # Find all EDF files
        edf_files = sorted(patient_dir.glob("*.edf"))
        
        if not edf_files:
            if self.verbose:
                print(f"  No EDF files found for {patient_id}")
            return
        
        subject_data = SubjectData(subject_id=patient_id)
        
        for edf_path in edf_files:
            file_name = edf_path.name
            file_name_normalized = normalize_filename(file_name)
            
            # Get seizures for this file - try multiple matching strategies
            seizure_info = annotations.get(file_name_normalized, {}).get("seizures", [])
            
            # Fallback: If no direct match, try matching by subject prefix
            # This handles cases like annotation says "PN01.edf" but file is "PN01-1.edf"
            if not seizure_info:
                subject_prefix = patient_id.lower()
                for ann_file, ann_data in annotations.items():
                    # Check if annotation file starts with subject ID (e.g., pn01)
                    ann_file_base = ann_file.replace('.edf', '').replace('-', '')
                    file_base = file_name_normalized.replace('.edf', '').replace('-', '')
                    
                    # Match if both start with same subject prefix
                    if ann_file_base.startswith(subject_prefix) and file_base.startswith(subject_prefix):
                        seizure_info = ann_data.get("seizures", [])
                        if seizure_info and self.verbose:
                            print(f"    [MATCH] {file_name} matched to annotations from {ann_file}")
                        break
            
            try:
                if self.load_data:
                    # Full loading - load data immediately
                    data, fs, channel_names = self._load_edf_file(str(edf_path))
                    recording_duration = data.shape[1] / fs
                    n_channels = data.shape[0]
                else:
                    # Lazy loading - get metadata only
                    recording_duration, n_channels, fs, channel_names = self._get_edf_metadata(str(edf_path))
                    data = None
                
                # Validate and clamp seizure times to recording duration
                valid_seizures = []
                for onset, offset in seizure_info:
                    if onset < 0:
                        onset = 0
                    if offset > recording_duration:
                        if self.verbose:
                            print(f"    Warning: {file_name} seizure end ({offset:.0f}s) > recording ({recording_duration:.0f}s), clamping")
                        offset = min(offset, recording_duration)
                    if onset >= recording_duration:
                        if self.verbose:
                            print(f"    Warning: {file_name} seizure start ({onset:.0f}s) >= recording ({recording_duration:.0f}s), skipping")
                        continue
                    if onset < offset:
                        valid_seizures.append(SeizureAnnotation(onset=onset, offset=offset))
                
                recording = Recording(
                    file_path=str(edf_path),
                    file_name=file_name,
                    subject_id=patient_id,
                    seizures=valid_seizures,
                    duration=recording_duration,
                    n_channels=n_channels,
                    fs=fs,
                    data=data,
                    channel_names=channel_names
                )
                
                subject_data.recordings.append(recording)
                
            except Exception as e:
                if self.verbose:
                    print(f"    Warning: Failed to load {file_name}: {e}")
        
        if subject_data.recordings:
            self.subjects[patient_id] = subject_data
            if self.verbose:
                print(f"  {patient_id}: {len(subject_data.recordings)} recordings, "
                      f"{subject_data.n_seizures} seizures")
    
    def _get_edf_metadata(self, file_path: str) -> Tuple[float, int, float, List[str]]:
        """Get EDF file metadata without loading data (memory efficient)."""
        if not HAS_MNE:
            raise ImportError("MNE is required. Install with: pip install mne")
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # preload=False means data is not loaded into memory
            raw = read_raw_edf(file_path, preload=False, verbose=False)
        
        duration = raw.n_times / raw.info['sfreq']
        n_channels = len(raw.ch_names)
        fs = raw.info['sfreq']
        channel_names = list(raw.ch_names)
        
        # Close the file handle
        del raw
        
        return duration, n_channels, fs, channel_names
    
    def load_recording_data(self, recording: Recording) -> np.ndarray:
        """Load data for a specific recording on-demand.
        
        Args:
            recording: Recording object to load data for
            
        Returns:
            EEG data array (n_channels, n_samples)
        """
        if recording.data is not None:
            return recording.data
        
        data, fs, channel_names = self._load_edf_file(recording.file_path)
        recording.data = data
        recording.fs = fs
        recording.channel_names = channel_names
        recording.n_channels = data.shape[0]
        recording.duration = data.shape[1] / fs
        
        return data
    
    def parse_summary_file(self, patient_id: str, patient_dir: Path) -> Dict[str, Dict]:
        """
        Parse patient summary file to extract seizure annotations.
        
        GenEEG pattern: handles Seizures-list-*.txt format.
        
        Returns:
            Dictionary mapping filename to seizure info:
            {'pn00-1.edf': {'seizures': [(start_sec, end_sec), ...]}}
        """
        # Find summary file - try multiple patterns
        summary_files = list(patient_dir.glob('Seizures-list-*.txt'))
        
        if not summary_files:
            summary_files = list(patient_dir.glob('*.summary'))
        
        if not summary_files:
            summary_files = list(patient_dir.glob('*.txt'))
        
        if not summary_files:
            if self.verbose:
                print(f"  Warning: No summary file found for {patient_id}")
            return {}
        
        seizure_data = {}
        current_file = None
        registration_start_seconds = None
        seizure_start_rel = None
        
        for summary_file in summary_files:
            if self.verbose:
                print(f"  Reading summary file: {summary_file.name}")
            
            with open(summary_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    
                    # Parse file name
                    if line.lower().startswith("file name:"):
                        current_file = normalize_filename(
                            line.split(":", 1)[1].strip()
                        )
                        seizure_data.setdefault(current_file, {"seizures": []})
                        registration_start_seconds = None
                        seizure_start_rel = None
                    
                    # Parse registration start time
                    elif line.lower().startswith("registration start time:") and current_file:
                        time_str = line.split(":", 1)[1].strip()
                        try:
                            registration_start_seconds = time_to_seconds(time_str)
                        except:
                            pass
                    
                    # Parse seizure start time
                    elif any(line.lower().startswith(s) for s in ["seizure start time:", "start time:"]) and current_file:
                        if registration_start_seconds is not None:
                            time_str = line.split(":", 1)[1].strip().replace(" ", "")
                            try:
                                seizure_start_abs = time_to_seconds(time_str)
                                seizure_start_rel = seizure_start_abs - registration_start_seconds
                                
                                # Handle day wrap-around
                                if seizure_start_rel < 0:
                                    seizure_start_rel += 24 * 3600
                            except:
                                pass
                    
                    # Parse seizure end time
                    elif any(line.lower().startswith(s) for s in ["seizure end time:", "end time:"]) and current_file:
                        if seizure_start_rel is not None and registration_start_seconds is not None:
                            time_str = line.split(":", 1)[1].strip().replace(" ", "")
                            try:
                                seizure_end_abs = time_to_seconds(time_str)
                                seizure_end_rel = seizure_end_abs - registration_start_seconds
                                
                                # Handle day wrap-around
                                if seizure_end_rel < seizure_start_rel:
                                    seizure_end_rel += 24 * 3600
                                
                                seizure_data[current_file]["seizures"].append(
                                    (seizure_start_rel, seizure_end_rel)
                                )
                                seizure_start_rel = None
                            except:
                                pass
        
        if self.verbose and seizure_data:
            total_seizures = sum(len(v["seizures"]) for v in seizure_data.values())
            print(f"    Found {len(seizure_data)} files with {total_seizures} total seizures")
        
        return seizure_data
    
    def _load_edf_file(self, file_path: str, preprocess: bool = True) -> Tuple[np.ndarray, float, List[str]]:
        """Load and preprocess a single EDF file."""
        if not HAS_MNE:
            raise ImportError("MNE is required. Install with: pip install mne")
        
        # Load with MNE
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            raw = read_raw_edf(file_path, preload=True, verbose=False)
        
        if preprocess:
            # Apply bandpass filter
            raw.filter(
                l_freq=self.bandpass_low,
                h_freq=self.bandpass_high,
                method='iir',
                iir_params={'order': 4, 'ftype': 'butter'},
                verbose=False
            )
            
            # Apply notch filter for power line noise
            if self.notch_freq:
                raw.notch_filter(
                    freqs=self.notch_freq,
                    method='fir',
                    fir_design='firwin',
                    verbose=False
                )
            
            # Resample if needed
            if abs(raw.info['sfreq'] - self.target_sr) > 0.1:
                raw.resample(self.target_sr, npad='auto', verbose=False)
        
        # Filter out non-EEG channels
        eeg_picks = []
        channel_names = raw.ch_names
        for i, name in enumerate(channel_names):
            if 'EKG' not in name.upper() and 'ECG' not in name.upper():
                eeg_picks.append(i)
        
        data = raw.get_data()[eeg_picks] if eeg_picks else raw.get_data()
        channel_names = [channel_names[i] for i in eeg_picks] if eeg_picks else channel_names
        sampling_rate = int(raw.info['sfreq'])
        
        return data, sampling_rate, channel_names
    
    def get_subject_ids(self) -> List[str]:
        """Get list of loaded subject IDs."""
        return list(self.subjects.keys())
    
    def get_subject_data(self, subject_id: str) -> SubjectData:
        """Get data for a specific subject."""
        if subject_id not in self.subjects:
            raise KeyError(f"Subject {subject_id} not found. Available: {self.get_subject_ids()}")
        return self.subjects[subject_id]


# ==============================================================================
# Dataset Factory
# ==============================================================================

def create_loader(data_dir: str, dataset_type: str = 'auto'):
    """
    Factory function to create appropriate loader.
    
    Parameters
    ----------
    data_dir : str
        Path to dataset directory
    dataset_type : str
        'chbmit', 'siena', or 'auto' (detect from directory name)
    
    Returns
    -------
    Loader instance (CHBMITLoader or SIENALoader)
    """
    data_path = Path(data_dir)
    
    if dataset_type == 'auto':
        name_lower = data_path.name.lower()
        if 'chb' in name_lower or 'mit' in name_lower:
            dataset_type = 'chbmit'
        elif 'siena' in name_lower or 'pn' in name_lower:
            dataset_type = 'siena'
        else:
            # Check contents
            if any(d.name.startswith('chb') for d in data_path.iterdir() if d.is_dir()):
                dataset_type = 'chbmit'
            elif any(d.name.startswith('PN') for d in data_path.iterdir() if d.is_dir()):
                dataset_type = 'siena'
            else:
                raise ValueError(f"Cannot auto-detect dataset type for: {data_dir}")
    
    if dataset_type == 'chbmit':
        return CHBMITLoader(data_dir)
    elif dataset_type == 'siena':
        return SIENALoader(data_dir)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


# ==============================================================================
# LOSO Split Generator
# ==============================================================================

def create_loso_splits(
    loader,
    holdout_subjects: Optional[List[str]] = None
) -> Iterator[Tuple[List[str], str]]:
    """
    Generate Leave-One-Subject-Out splits.
    
    Parameters
    ----------
    loader : CHBMITLoader or SIENALoader
        Loaded dataset
    holdout_subjects : List[str], optional
        Subjects to exclude from LOSO (reserved for final testing)
    
    Yields
    ------
    train_subjects : List[str]
        Subject IDs for training
    test_subject : str
        Subject ID for testing
    """
    all_subjects = loader.get_subject_ids()
    
    if holdout_subjects:
        loso_subjects = [s for s in all_subjects if s not in holdout_subjects]
    else:
        loso_subjects = all_subjects
    
    for test_subject in loso_subjects:
        train_subjects = [s for s in loso_subjects if s != test_subject]
        yield train_subjects, test_subject
