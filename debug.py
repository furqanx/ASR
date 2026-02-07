import os
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
import scipy.io.wavfile as wav
import shutil

from src.utils.file_utils import collect_filepaths_and_labels
from src.utils.sampler import (
    BalancedBatchScheduler,
    TorchBalancedSampler
)

from src.preprocessing import load_audio

from src.dataloader import AudioTFDataset, AudioPTDataset

#######################################################
import librosa
import soundfile
import tensorflow as tf
import numpy as np
import noisereduce as nr
from glob import glob
import os

from spafe.features.cqcc import cqcc
from spafe.features.cqcc import cqt_spectrogram
from spafe.features.gfcc import gfcc
from spafe.features.gfcc import erb_spectrogram
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
from spafe.utils.preprocessing import framing, windowing
from spafe.utils.preprocessing import pre_emphasis, framing, windowing, SlidingWindow

from tqdm import tqdm

import torch
import torchaudio
# import torchaudio.functional as F
from torchaudio.transforms import MFCC
from torchaudio.transforms import MelSpectrogram
from scipy.interpolate import interp1d
from scipy.io.wavfile import read
from scipy.fftpack import dct

import torch
import torchaudio
import torch.nn.functional as F

import librosa
import soundfile

from scipy.io.wavfile import read
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features
from scipy.io.wavfile import read
from spafe.features.mfcc import mfcc, mel_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features
from spafe.features.mfcc import mel_spectrogram
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from scipy.io.wavfile import read
from spafe.features.cqcc import cqt_spectrogram
from spafe.utils.vis import show_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from scipy.io.wavfile import read
#######################################################







# --- 1. SETUP DUMMY DATA -----------------------------------
print("[1] Generating Dummy Data...")
DUMMY_DIR = "dummy_dataset"
if os.path.exists(DUMMY_DIR):
    shutil.rmtree(DUMMY_DIR)
os.makedirs(DUMMY_DIR)

# Kita buat 4 Speaker (Class 0, 1, 2, 3), masing-masing 5 file audio
dummy_paths = []
dummy_labels = []
n_speakers = 4
n_files_per_speaker = 5

for spk_id in range(n_speakers):
    for i in range(n_files_per_speaker):
        fname = os.path.join(DUMMY_DIR, f"spk{spk_id}_{i}.wav")
        # Buat audio sine wave sederhana
        sr = 16000
        t = np.linspace(0, 4, sr * 4) # 4 detik
        y = 0.5 * np.sin(2 * np.pi * (220 + spk_id*50) * t)
        wav.write(fname, sr, y.astype(np.float32))
        
        dummy_paths.append(fname)
        dummy_labels.append(spk_id)

print(f"    Total files: {len(dummy_paths)}")
print(f"    Labels sample: {dummy_labels[:10]}...")

# Config sederhana
config = {
    'sample_rate': 16000,
    'target_duration': 3,
    'feature_extractors': ['gfcc'] # Asumsi Anda punya fungsi ini atau mock
}


# --- 2. TEST SAMPLER LOGIC ---------------------------------
print("\n[2] Testing BalancedBatchScheduler (Logic Core)...")
# Skenario DML: 2 Speaker per batch, 2 Utterance per speaker
# Total batch size = 4
P, K = 2, 2 
scheduler = BalancedBatchScheduler(dummy_labels, n_classes=P, n_samples=K)

print(f"    Skenario: P={P}, K={K} (Batch Size {P*K})")
for i, batch_indices in enumerate(scheduler):
    batch_labels = [dummy_labels[idx] for idx in batch_indices]
    print(f"    Batch {i} Indices: {batch_indices}")
    print(f"    Batch {i} Labels : {batch_labels}")
    
    # Validasi Logic
    if len(set(batch_labels)) != P:
        print("    [FAIL] Jumlah speaker dalam batch salah!")
    elif batch_labels[0] != batch_labels[1]: # Cek pasangan (K=2)
        print("    [FAIL] Pasangan label tidak berurutan!")
    else:
        print("    [PASS] Logika Sampler OK.")
    break # Cek 1 batch saja






# --- 3. TEST TENSORFLOW PIPELINE ---------------------------
print("\n[3] Testing TensorFlow DML Pipeline...")
try:
    tf_ds = AudioTFDataset(file_paths=dummy_paths, labels=dummy_labels)
    # Panggil method khusus DML
    dml_tf_dataset = tf_ds.get_dml_dataset(n_classes=P, n_samples=K)
    
    for features, labels in dml_tf_dataset.take(1):
        print("    [PASS] Batch Loaded via TensorFlow.")
        
        # Cek kunci apa saja yang tersedia (DEBUGGING PENTING)
        print(f"    Available Keys: {list(features.keys())}")
        
        # print(f"    Label Shape: {labels.shape} (Expect: {P*K})")
        # print(f"    Label Data : {labels.numpy()}")
        
        if 'gfcc' in features:
            print(f"    GFCC Shape : {features['gfcc'].shape}") # (Batch, Time, Freq, Ch)    
            # Cek Tipe Data
            if features['gfcc'].dtype == tf.float32:
                print("    [PASS] Data Type is float32.")
            else:
                print(f"    [FAIL] Data Type is {features['gfcc'].dtype}")
        else:
            print("    [WARNING] Key 'gfcc' tidak ditemukan di output dataset!")

except Exception as e:
    print(f"    [FAIL] Error TF: {e}")
    import traceback
    traceback.print_exc()


# --- 4. TEST PYTORCH PIPELINE ------------------------------
print("\n[4] Testing PyTorch DML Pipeline...")
try:
    pt_ds = AudioPTDataset(file_paths=dummy_paths, label=dummy_labels)
    
    # Setup Sampler Khusus
    torch_sampler = TorchBalancedSampler(dummy_labels, n_classes=P, n_samples=K)
    
    # Setup DataLoader
    # loader = DataLoader(pt_ds, batch_sampler=None, sampler=torch_sampler, batch_size=P*K)
    loader = DataLoader(
        pt_ds, 
        batch_sampler=None,
        sampler=torch_sampler,
        batch_size=P*K,
        drop_last=True
    )
    
    data_iter = iter(loader)
    features, labels = next(data_iter)
    
    print("    [PASS] Batch Loaded via PyTorch.")
    print(f"    Available Keys: {list(features.keys())}") # Debug keys
    print(f"    Label Shape: {labels.shape} (Expect: {P*K})")
    print(f"    Label Data : {labels}")
    
    if 'gfcc' in features:
        print(f"    GFCC Shape : {features['gfcc'].shape}")
        
        if features['gfcc'].dtype == torch.float32:
            print("    [PASS] Data Type is float32.")
        else:
            print(f"    [FAIL] Data Type is {features['gfcc'].dtype}")
    else:
            print("    [WARNING] Key 'gfcc' tidak ditemukan di output dataset!")

except Exception as e:
    print(f"    [FAIL] Error PT: {e}")
    import traceback
    traceback.print_exc()