import os
import numpy as np
import noisereduce as nr
import librosa
import tensorflow as tf
import torch

# ==========================================
#         CORE FUNCTIONS (NumPy Based)
# ==========================================

def _load_audio(file_name, target_sr):
    """Core logic: Load audio ke Numpy Array"""
    if isinstance(file_name, bytes):
        file_name = file_name.decode('utf-8')
    elif isinstance(file_name, tf.Tensor) and file_name.dtype == tf.string:
        file_name = file_name.numpy().decode('utf-8')
    
    try:
        waveform_np, _ = librosa.load(file_name, sr=target_sr, mono=True)
        return waveform_np.astype(np.float32)
    except Exception as e:
        print(f"WARNING: Gagal memuat {file_name}. Error: {e}")
        return np.zeros(1, dtype=np.float32)

def _trim_audio(waveform_np, top_db=20):
    """Core logic: Trim silence"""
    trimmed, _ = librosa.effects.trim(waveform_np, top_db=top_db)
    return trimmed

def _reduce_noise(waveform_np, sample_rate=16000):
    """Core logic: Noise reduction"""
    return nr.reduce_noise(y=waveform_np, sr=sample_rate, prop_decrease=1.0)

def _normalize(waveform_np):
    """Core logic: Normalize to [-1, 1]"""
    max_val = np.max(np.abs(waveform_np))
    if max_val > 0:
        return waveform_np / max_val
    return waveform_np

def _segment(waveform_np, target_length):
    """Core logic: Pad or Slice"""
    current_length = waveform_np.shape[0]
    
    if current_length > target_length:
        return waveform_np[:target_length]
    elif current_length < target_length:
        pad_width = target_length - current_length
        return np.pad(waveform_np, (0, pad_width), mode='constant')
    
    return waveform_np


def _pitch_shift(waveform_np, sample_rate, n_steps=2):
    # Menggeser nada suara (augmentasi)
    return librosa.effects.pitch_shift(waveform_np, sr=sample_rate, n_steps=n_steps)


# ==========================================
#         WRAPPERS (Framework Specific)
# ==========================================

def load_audio(file_name, target_sr, backend='tf'):
    """
    Args:
        file_name: path audio
        backend: 'tf' (TensorFlow) atau 'pt' (PyTorch) atau 'np' (Numpy)
    """
    # 1. Jalankan Core Logic
    waveform_np = _load_audio(file_name, target_sr)
    
    # 2. Konversi sesuai backend
    if backend == 'tf':
        return tf.convert_to_tensor(waveform_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(waveform_np).float()
    else:
        return waveform_np

def silence_trimming(waveform, backend='tf', top_db=20):
    # Pastikan input jadi numpy dulu
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform
        
    trimmed_np = _trim_audio(wav_np, top_db)
    
    if backend == 'tf':
        return tf.convert_to_tensor(trimmed_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(trimmed_np).float()
    return trimmed_np

def reduce_noise(waveform, backend='tf', sample_rate=16000):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    reduced_noise_np = _reduce_noise(wav_np, sample_rate=sample_rate)

    if backend == 'tf':
        return tf.convert_to_tensor(reduced_noise_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(reduced_noise_np).float()
    return reduced_noise_np

def normalize_audio(waveform, backend='tf'):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    normalized_np = _normalize(wav_np)

    if backend == 'tf':
        return tf.convert_to_tensor(normalized_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(normalized_np).float()
    return normalized_np

def segment_audio(waveform, target_length, backend='tf'):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    segmented_np = _segment(wav_np, target_length)

    if backend == 'tf':
        return tf.convert_to_tensor(segmented_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(segmented_np).float()
    return segmented_np


def apply_pitch_shift(waveform, backend='tf', sample_rate=16000, n_steps=2):
    if backend == 'tf' and isinstance(waveform, tf.Tensor):
        wav_np = waveform.numpy()
    elif backend == 'pt' and isinstance(waveform, torch.Tensor):
        wav_np = waveform.numpy()
    else:
        wav_np = waveform

    shifted_np = _pitch_shift(wav_np, sample_rate, n_steps)

    if backend == 'tf':
        return tf.convert_to_tensor(shifted_np, dtype=tf.float32)
    elif backend == 'pt':
        return torch.from_numpy(shifted_np).float()
    return shifted_np