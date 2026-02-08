import os
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm

# Import fungsi preprocessing berat Anda
# Pastikan path import ini sesuai dengan struktur folder Anda
from src.preprocessing import load_audio, reduce_noise, silence_trimming

def preprocess_offline(source_dir, target_dir, sample_rate=16000):
    """
    Membaca data mentah, membersihkan (trim/denoise), dan menyimpan ke folder baru.
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    print(f"🚀 Memulai Preprocessing Offline...")
    print(f"📂 Source: {source_path}")
    print(f"📂 Target: {target_path}")

    # Ambil semua file .wav secara rekursif
    files = list(source_path.rglob('*.wav'))
    
    # Filter: Jangan proses folder background_noise jika ikut termuat
    # Kita butuh background noise tetap murni/raw untuk augmentasi nanti
    # files = [f for f in files if "_background_noise_" not in str(f)]

    success_count = 0
    
    for file_path in tqdm(files, desc="Processing"):
        try:
            # 1. Load Audio
            waveform = load_audio(str(file_path), backend='pt', target_sr=sample_rate)
            
            # ====================================================
            # HEAVY LIFTING ZONE (Proses Berat Di Sini)
            # ====================================================
            
            # 2. Silence Trimming (Wajib Offline)
            # Buang bagian hening di awal/akhir agar file lebih padat
            waveform = silence_trimming(waveform, backend='pt', top_db=20)
            
            # 3. Reduce Noise (Opsional - SANGAT BERAT)
            waveform = reduce_noise(waveform, backend='pt', sample_rate=sample_rate)
            
            # ====================================================
            
            # 4. Simpan ke Folder Baru
            # Kita pertahankan struktur folder (misal: train/yes/01.wav)
            relative_path = file_path.relative_to(source_path)
            save_path = target_path / relative_path
            
            # Buat folder jika belum ada
            os.makedirs(save_path.parent, exist_ok=True)
            
            # Simpan
            torchaudio.save(str(save_path), waveform.unsqueeze(0), sample_rate)
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

    print(f"\n✅ Selesai! {success_count} file berhasil diproses.")
    print(f"📁 Dataset bersih tersimpan di: {target_path}")

if __name__ == "__main__":
    # KONFIGURASI PATH
    
    # Sesuaikan dengan lokasi dataset mentah Anda sekarang
    RAW_DATA_PATH = "LAB-AI/dataset_kws_split" 
    
    # Lokasi tujuan dataset bersih
    PROCESSED_DATA_PATH = "LAB-AI/dataset_kws_processed"
    
    preprocess_offline(RAW_DATA_PATH, PROCESSED_DATA_PATH)