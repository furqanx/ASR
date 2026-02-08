import os
from glob import glob
import numpy as np

try:
    from spafe.features.cqcc import cqcc
    from spafe.features.gfcc import gfcc
except ImportError:
    print("[Warning] spafe not installed. Feature extraction might fail")

import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

from .preprocessing import (
    load_audio, 
    reduce_noise, 
    normalize_audio, 
    silence_trimming, 
    segment_audio,
    load_noise_files,
    add_background_noise
)

from .utils.sampler import (
    BalancedBatchScheduler, 
    TorchBalancedSampler
)

class AudioTFDataset:
    def __init__(self, file_paths, labels, feature_extractors=['gfcc']):
        self.file_paths = file_paths
        self.labels = labels
        self.feature_extractors = feature_extractors
        self.sample_rate = 16000
        self.target_duration = 3
        
    def __processing__(self, file_path_tensor, label_tensor):
        """
        Ini adalah logika Python murni (mirip __getitem__ di PyTorch).
        Fungsi ini akan dijalankan di CPU.
        """
        # 1. Decode byte string dari Tensor ke string path biasa
        file_name = file_path_tensor.numpy().decode('utf-8')
        label = label_tensor.numpy()
        
        # 1. Load data
        waveform = load_audio(file_name, backend='tf') # Return numpy array/list

        # 2. Noise reduction
        waveform = reduce_noise(waveform, backend='tf', sample_rate=self.sample_rate)
        
        # 3. Normalisasi
        waveform = normalize_audio(waveform, backend='tf')
        
        # 4. Smart trimming
        waveform = silence_trimming(waveform, backend='tf', top_db=20)

        # 5. Segmentasi
        target_length = int(self.target_duration * self.sample_rate)
        waveform = segment_audio(waveform, backend='tf', target_length=target_length)
        
        # Pastikan format numpy float32
        waveform_np = np.array(waveform).astype(np.float32)
        # Reshape (N, 1)
        waveform_reshaped = waveform_np[:, np.newaxis]


        ############################
        # --- FEATURE EXTRACTION ---
        ############################
        # Kita harus return array fix, bukan dict dinamis agar mudah ditangkap TF
        # Misal kita asumsikan output GFCC shape-nya (Time, Freq, 1)
        gfcc_feat = np.zeros((1, 1), dtype=np.float32) # Placeholder
        cqcc_feat = np.zeros((1, 1), dtype=np.float32) # Placeholder
        
        # Logika ekstraksi
        if 'gfcc' in self.feature_extractors:
            gfcc_feat = gfcc(sig=waveform_reshaped).astype(np.float32)
        
        if 'cqcc' in self.feature_extractors:
            cqcc_feat = cqcc(sig=waveform_reshaped).astype(np.float32)

        return gfcc_feat, cqcc_feat, label


    def get_dataset(self):
        """
        Membangun tf.data.Dataset
        """
        # 1. Buat dataset dasar dari path dan label
        dataset = tf.data.Dataset.from_tensor_slices((self.file_paths, self.labels))

        # 2. Bungkus fungsi python kita dengan tf.py_function
        def tf_wrapper(file_path, label):
            # tf.py_function(func, inputs, output_types)
            gfcc_out, cqcc_out, label_out = tf.py_function(
                func=self.__processing__,

                inp=[file_path, label],
                
                # Tentukan tipe data outputnya
                Tout=[tf.float32, tf.float32, tf.int64]
            )
            
            # PENTING: Set shape output (tf.py_function menghilangkan info shape)
            # Sesuaikan shape ini dengan output ekstraktor fitur Anda
            # Contoh: (Time, Feature_Dim)
            gfcc_out.set_shape([None, None]) 
            cqcc_out.set_shape([None, None])
            label_out.set_shape([]) # Scalar
            
            # Kembalikan dictionary agar rapi seperti PyTorch code Anda
            features = {}
            if 'gfcc' in self.feature_extractors:
                features['gfcc'] = gfcc_out
            if 'cqcc' in self.feature_extractors:
                features['cqcc'] = cqcc_out
                
            return features, label_out

        # 3. Map wrapper ke dataset
        # num_parallel_calls=tf.data.AUTOTUNE agar proses load berjalan paralel (multiprocessing)
        dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset


    def get_dml_dataset(self, n_classes=4, n_samples=2):
        """
        Versi Dataset KHUSUS untuk Deep Metric Learning
        """
        # 1. Inisialisasi Scheduler
        scheduler = BalancedBatchScheduler(self.labels, n_classes, n_samples)
        
        # 2. Buat Generator Python
        def generator():
            # Scheduler menghasilkan list index per batch
            for batch_indices in scheduler: 
                for idx in batch_indices:
                    # Yield data satu per satu sesuai urutan scheduler
                    yield self.file_paths[idx], self.labels[idx]

        # 3. Buat Dataset dari Generator
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int64)
            )
        )

        # 4. Map Processing (Sama seperti sebelumnya)
        # Gunakan wrapper self.__processing__ yang sudah ada
        def tf_wrapper(file_path, label):
            # tf.py_function(func, inputs, output_types)
            gfcc_out, cqcc_out, label_out = tf.py_function(
                func=self.__processing__,

                inp=[file_path, label],
                
                # Tentukan tipe data outputnya
                Tout=[tf.float32, tf.float32, tf.int64]
            )
            
            # PENTING: Set shape output (tf.py_function menghilangkan info shape)
            # Sesuaikan shape ini dengan output ekstraktor fitur Anda
            # Contoh: (Time, Feature_Dim)
            gfcc_out.set_shape([None, None]) 
            cqcc_out.set_shape([None, None])
            label_out.set_shape([]) # Scalar
            
            # Kembalikan dictionary agar rapi seperti PyTorch code Anda
            features = {}
            if 'gfcc' in self.feature_extractors:
                features['gfcc'] = gfcc_out
            if 'cqcc' in self.feature_extractors:
                features['cqcc'] = cqcc_out
                
            return features, label_out

        dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        
        # 5. Batching
        # PENTING: Batch size harus pas P*K. 
        # Karena generator sudah mengurutkan P*K, batching disini tinggal "memotong" stream.
        dataset = dataset.batch(n_classes * n_samples)
        
        return dataset

class AudioPTDataset(Dataset):
    def __init__(self, 
        file_paths, labels, 
        preprocess=True, target_duration=3.0, sample_rate=16_000, 
        feature_extractors=[], background_noise=False, noise_path=None):
        self.file_paths = file_paths
        self.labels = labels
        self.preprocess_enabled = preprocess
        self.feature_extractors = feature_extractors
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.background_noise = background_noise

        self.noise_dict = {}
        if self.background_noise and noise_path:
            # Load noise sekali saja saat inisialisasi dataset
            # Agar RAM efisien dan tidak baca harddisk terus menerus
            print("[Dataset] Loading background noise files...")
            self.noise_dict = load_noise_files(noise_path, sample_rate=self.sample_rate)
            
            if not self.noise_dict:
                print("[Warning] Background noise diaktifkan tapi folder kosong/tidak ditemukan. Augmentasi akan dilewati.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Dapatkan info file dan label
        file_name = self.file_paths[idx]
        label = self.labels[idx]

        # 1. Load data
        waveform = load_audio(file_name, backend='pt', target_sr=self.sample_rate)

        if self.preprocess_enabled:
            # 2. Preprocessing: hilangkan silent
            waveform = silence_trimming(waveform, backend='pt', top_db=20)
            # 3. Noise reduction 
            waveform = reduce_noise(waveform, backend='pt', sample_rate=self.sample_rate)

        if self.background_noise and self.noise_dict:
            # A. Konversi Tensor ke Numpy (karena fungsi add_background_noise pakai numpy)
            wav_numpy = waveform.numpy()
            # Jika dimensi (1, N), ubah jadi (N,) untuk pemrosesan numpy
            if wav_numpy.ndim > 1:
                wav_numpy = wav_numpy.squeeze()
                
            # B. Tambahkan Noise
            # noise_reduction bisa di-randomize juga kalau mau variatif
            wav_noisy = add_background_noise(wav_numpy, self.noise_dict, noise_reduction=0.5)
            
            # C. Kembalikan ke Tensor PyTorch
            waveform = torch.from_numpy(wav_noisy)
            
            # Kembalikan dimensi channel jika hilang (jadi [1, N])
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

        if self.preprocess_enabled:
            # 4. Normalisasi amplitudo ke [-1,1]
            waveform = normalize_audio(waveform, backend='pt')

        # 5. Standarisasi durasi (proses Segmentasi)
        target_length = int(self.target_duration * self.sample_rate)
        waveform = segment_audio(waveform, backend='pt', target_length=target_length)

        # KASUS A: RAW AUDIO (Jika feature_extractors kosong)
        # Digunakan untuk: RawNet, ResNetSE, ECAPA-TDNN
        if not self.feature_extractors or len(self.feature_extractors) == 0:
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
            return waveform, torch.tensor(label).long()
        # KASUS B: FEATURE EXTRACTION (Jika feature_extractors ada isinya)
        # Digunakan untuk: Model klasik / Eksperimen fitur manual
        else:
            # Konversi pytorch ke numpy
            waveform_np = waveform.numpy()
            # Spafe butuh input (N, 1) atau (N,) tergantung versi, amannya kita reshape
            if waveform_np.ndim == 1:
                waveform_reshaped = waveform_np[:, np.newaxis]
            else:
                waveform_reshaped = waveform_np

            # 6. Ekstraksi fitur
            features = {}

            # Ekstraksi GFCC
            if 'gfcc' in self.feature_extractors:
                # Konversi hasil numpy ke Tensor PyTorch
                feat = gfcc(sig=waveform_reshaped, fs=self.sample_rate)
                features['gfcc'] = torch.from_numpy(feat).float()
                
            # Ekstraksi CQCC
            if 'cqcc' in self.feature_extractors:
                feat = cqcc(sig=waveform_reshaped, fs=self.sample_rate)
                features['cqcc'] = torch.from_numpy(feat).float()

        return features, torch.tensor(label).long()
    
def create_dataset_from_path(data_path, config):
    """Helper function untuk membuat instance Dataset dari path folder"""
    files = glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    if not files:
        return None

    # Labeling berdasarkan nama folder parent
    classes = sorted(list(set([os.path.basename(os.path.dirname(f)) for f in files])))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = [class_to_idx[os.path.basename(os.path.dirname(f))] for f in files]
    
    feats_to_use = config.get('features', [])

    ds = AudioPTDataset(
        file_paths=files, 
        labels=labels, 
        preprocess=True,
        feature_extractors=feats_to_use,
        target_duration=config.get('max_duration', 3.0)
    )

    return ds

def get_dataloader(config):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 2)

    # --- 1. SETUP TRAIN LOADER ---
    train_ds = create_dataset_from_path(config['train_path'], config)
    if train_ds is None:
        raise RuntimeError(f"No audio files found in train path: {config['train_path']}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,           # Shuffle untuk Training
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # --- 2. SETUP VAL LOADER ---
    val_loader = None
    if config.get('val_path'):
        val_ds = create_dataset_from_path(config['val_path'], config)
        
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds, 
                batch_size=batch_size, 
                shuffle=False,      # Jangan shuffle Validasi
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False     # Jangan buang sisa data validasi
            )
            print(f"[DataLoader] Validation Loader created with {len(val_ds)} samples.")
        else:
            print(f"[DataLoader] Warning: val_path provided but no files found.")

    return train_loader, val_loader