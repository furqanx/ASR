import sys
import torch

# --- 1. IMPORT ARSITEKTUR PYTORCH ---
# Kita mengimpor fungsi 'MainModel' dari masing-masing file dan memberinya alias
# agar tidak bentrok (karena namanya sama-sama MainModel).
try:
    from src.models.pytorch.resnetse34l import MainModel as ResNetSE34L
    from src.models.pytorch.resnetse34v2 import MainModel as ResNetSE34V2
    from src.models.pytorch.rawnet3 import MainModel as RawNet3
    from src.models.pytorch.vggvox import MainModel as VGGVox
except ImportError as e:
    print(f"[Factory Error] Gagal mengimport model PyTorch: {e}")
    # Error ini mungkin muncul jika library dependensi (spt asteroid-filterbanks) belum diinstall
    pass

def build_model(config):
    """
    Factory Function: Membangun model berdasarkan konfigurasi.
    
    Args:
        config (dict): Dictionary konfigurasi (biasanya bagian 'model' dari config.yaml). Wajib memiliki key 'name'.
    
    Returns:
        model (nn.Module): Object model yang siap digunakan.
    """
    
    # 1. Cek Backend (PyTorch vs TensorFlow)
    # Default ke 'pytorch' jika tidak didefinisikan
    backend = config.get('backend', 'pytorch').lower()
    
    if backend == 'pytorch':
        return _build_pytorch_model(config)
    elif backend == 'tensorflow':
        raise NotImplementedError("Backend TensorFlow belum diimplementasikan di factory ini.")
    else:
        raise ValueError(f"Backend '{backend}' tidak didukung. Gunakan 'pytorch' atau 'tensorflow'.")

def _build_pytorch_model(config):
    """Internal function untuk menangani model PyTorch"""
    
    # Ambil nama model dari config, default ke 'resnetse34l' jika kosong
    model_name = config.get('name', 'resnetse34l').lower()
    
    print(f"[Model Factory] Initializing {model_name}...")

    # --- ROUTING LOGIC ---
    if model_name == 'resnetse34l':
        # ResNetSE-34 (Versi Light/Lama - Filter mulai 16)
        model = ResNetSE34L(**config)
        
    elif model_name == 'resnetse34v2':
        # ResNetSE-34 (Versi V2 - Filter mulai 32, lebih besar)
        model = ResNetSE34V2(**config)
        
    elif model_name == 'rawnet3':
        # RawNet3 (End-to-End Raw Audio)
        # RawNet butuh argumen 'embedding_dim' yang diterjemahkan wrapper jadi 'nOut'
        model = RawNet3(**config)
        
    elif model_name == 'vggvox':
        # VGGVox (Model Klasik untuk Benchmark)
        model = VGGVox(**config)
        
    else:
        raise ValueError(f"Nama model '{model_name}' tidak dikenal. "
                        f"Pilihan: resnetse34l, resnetse34v2, rawnet3, vggvox")

    return model