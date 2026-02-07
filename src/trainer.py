import os
import time
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import torch
import tensorflow as tf
from tqdm import tqdm

from src.utils.metrics import compute_metrics_report

class BaseTrainer(ABC):
    """
    BaseTrainer adalah kerangka abstrak untuk melatih model.
    Kelas ini menangani:
    1. Loop utama (fit)
    2. Logging (mencatat loss/metric)
    3. Manajemen folder output (checkpoints)
    4. Saving/Loading konfigurasi
    """
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 1. Setup Direktori Output
        # Format: experiments/NAMA_MODEL_TIMESTAMP (agar tidak menimpa training lama)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = config.get("model_name", "unnamed_model")
        self.output_dir = os.path.join("logs", "experiments", f"{model_name}_{timestamp}")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "checkpoints"))
            
        # 2. Setup Logger (Print ke layar & simpan ke file log.txt)
        self.logger = self._setup_logger()
        
        # 3. Simpan Config untuk Reproducibility
        self._save_config()
        
        # 4. Variabel Tracking
        self.start_epoch = 0
        self.best_loss = float('inf') # Untuk tracking model terbaik
        self.best_metric = -float('inf') # Misal akurasi/EER terbaik
        
        self.logger.info(f"Experiment initialized at: {self.output_dir}")

    def _setup_logger(self):
        """Membuat logger agar output tercatat di terminal dan file log"""
        logger = logging.getLogger("Trainer")
        logger.setLevel(logging.INFO)
        
        # Handler File
        fh = logging.FileHandler(os.path.join(self.output_dir, "train_log.txt"))
        fh.setLevel(logging.INFO)
        
        # Handler Console (Terminal)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def _save_config(self):
        """Menyimpan dictionary config ke JSON"""
        config_path = os.path.join(self.output_dir, "config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            self.logger.warning(f"Failed to save config: {e}")

    def fit(self, epochs):
        """
        Loop Utama Training (The Master Loop)
        """
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()
            
            # --- 1. Training Phase ---
            self.logger.info(f"=== Epoch {epoch+1}/{epochs} ===")
            train_metrics = self.train_epoch(epoch)
            
            # --- 2. Validation Phase ---
            val_metrics = self.valid_epoch(epoch)
            
            # --- 3. Logging ---
            duration = time.time() - start_time
            self._log_epoch(epoch, train_metrics, val_metrics, duration)
            
            # --- 4. Checkpoint Strategy ---
            # Simpan Last Model (selalu update)
            self.save_checkpoint(os.path.join(self.output_dir, "checkpoints", "last_model.pth"))
            
            # Simpan Best Model (hanya jika loss turun)
            # Asumsi: val_metrics punya key 'loss'
            current_val_loss = val_metrics.get('loss', float('inf'))
            if current_val_loss < self.best_loss:
                self.best_loss = current_val_loss
                self.logger.info(f"New best model found! (Score/EER: {self.best_loss:.4f})")
                self.save_checkpoint(os.path.join(self.output_dir, "checkpoints", "best_model.pth"))

    def _log_epoch(self, epoch, train_metrics, val_metrics, duration):
        """Helper untuk print hasil epoch dengan rapi"""
        msg = f"Epoch {epoch+1} completed in {duration:.0f}s | "
        
        # Format metrics training
        for k, v in train_metrics.items():
            msg += f"Train {k}: {v:.4f} | "
            
        # Format metrics validation
        for k, v in val_metrics.items():
            msg += f"Val {k}: {v:.4f} | "
            
        self.logger.info(msg)

    # --- ABSTRACT METHODS (Wajib diisi oleh TorchTrainer / TFTrainer) ---
    
    @abstractmethod
    def train_epoch(self, epoch):
        """
        Logika satu epoch training.
        Harus return dictionary metrics, misal: {'loss': 0.5, 'acc': 0.9}
        """
        pass

    @abstractmethod
    def valid_epoch(self, epoch):
        """
        Logika satu epoch validasi.
        Harus return dictionary metrics.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path):
        """Logika menyimpan bobot model ke disk"""
        pass

    @abstractmethod
    def load_checkpoint(self, path):
        """Logika memuat bobot model dari disk"""
        pass


# --- Tambahkan library PyTorch di bagian paling atas file jika belum ada ---
import torch
from tqdm import tqdm # Library untuk progress bar

# ... (Kode BaseTrainer Anda sebelumnya ada di sini) ...

class TorchTrainer(BaseTrainer):
    """
    Implementasi Trainer khusus untuk PyTorch.
    """
    def __init__(self, model, train_loader, val_loader, config, optimizer, criterion, device=None, scheduler=None):
        # Panggil init milik BaseTrainer
        super().__init__(model, train_loader, val_loader, config)
        
        self.optimizer = optimizer
        self.criterion = criterion # Loss function (CrossEntropy / TripletLoss)
        self.scheduler = scheduler # Learning Rate Scheduler (Opsional)
        
        # Auto-detect GPU jika device tidak diset manual
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pindahkan model ke GPU/CPU
        self.model = self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

    def train_epoch(self, epoch):
        self.model.train() # Set mode ke Training (aktifkan Dropout/BatchNorm)
        total_loss = 0
        
        # Gunakan TQDM untuk progress bar visual
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", unit="batch")
        
        for batch_idx, (data, target) in enumerate(pbar):
            # 1. Pindahkan data ke Device (GPU)
            data, target = data.to(self.device), target.to(self.device)
            
            # 2. Reset Gradient (Wajib di PyTorch sebelum backprop)
            self.optimizer.zero_grad()
            
            # 3. Forward Pass (Jalan ke depan)
            output = self.model(data)
            
            # 4. Hitung Error (Loss)
            loss = self.criterion(output, target)
            
            # 5. Backward Pass (Hitung gradien)
            loss.backward()
            
            # 6. Update Bobot
            self.optimizer.step()
            
            # Logging progress bar
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(self.train_loader)
        
        # Update Learning Rate jika pakai Scheduler
        if self.scheduler:
            self.scheduler.step()
            
        return {'loss': avg_loss}

    def valid_epoch(self, epoch):
        """
        Melakukan validasi dengan menghitung EER (Equal Error Rate).
        Karena ini Open-Set Speaker Recognition, kita tidak menghitung Loss ArcFace,
        melainkan menghitung seberapa mirip embedding antar-speaker.
        """
        self.model.eval() # Mode Evaluasi (Matikan Dropout, Freeze BatchNorm)
        
        # Container untuk menampung seluruh data validasi
        all_embeddings = []
        all_labels = []
        
        # Setup Progress Bar
        pbar = tqdm(self.val_loader, desc=f"Valid Epoch {epoch+1}", unit="batch")
        
        with torch.no_grad(): # Hemat memori, matikan gradien
            for batch_idx, (data, target) in enumerate(pbar):
                # 1. Pindah ke GPU
                data = data.to(self.device)
                
                # 2. Forward Pass (Ambil Embedding, bukan Logits)
                # Pastikan model Anda return embedding vector di tahap ini
                embeddings = self.model(data)
                
                # 3. Simpan ke CPU (agar GPU tidak penuh)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(target.cpu())

        # --- TAHAP PERHITUNGAN METRIK ---
        
        # 1. Gabungkan list menjadi satu Tensor besar
        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)
        
        # 2. Konversi ke Numpy Array (karena metrics.py pakai Scikit-Learn/Numpy)
        embeddings_np = all_embeddings.numpy()
        labels_np = all_labels.numpy()
        
        # 3. Hitung EER menggunakan fungsi dari metrics.py
        # Mode 'dml' akan memicu pemanggilan compute_eer
        metrics = compute_metrics_report(embeddings_np, labels_np, mode='dml')
        
        eer_score = metrics['eer']
        threshold = metrics['threshold']
        
        print(f"\n[Validation] EER: {eer_score:.4f} | Threshold: {threshold:.4f}")

        # --- PENTING UNTUK SAVE CHECKPOINT ---
        # BaseTrainer biasanya menyimpan 'best_model' jika 'loss' turun.
        # Karena kita ingin menyimpan model jika EER turun, kita "tipu" sedikit:
        # Kita masukkan nilai EER ke dalam key 'loss'.
        
        return {
            'loss': eer_score,      # Trik agar BaseTrainer menyimpan model saat EER turun
            'val_loss': eer_score,  # Sama, untuk display
            'threshold': threshold
        }

    def save_checkpoint(self, path):
        """Simpan state model, optimizer, dan config"""
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'epoch': self.start_epoch
        }
        torch.save(state, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load state model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Checkpoint loaded successfully from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")

class TFTrainer(BaseTrainer):
    """
    Implementasi Trainer khusus untuk TensorFlow.
    Menggunakan Custom Training Loop dengan tf.GradientTape.
    """
    def __init__(self, model, train_loader, val_loader, config, optimizer, loss_fn):
        super().__init__(model, train_loader, val_loader, config)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        # Metrics tracker (untuk merata-ratakan loss per epoch)
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')

    @tf.function # Magic Decorator: Mengompilasi fungsi ini jadi Graph (Cepat!)
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            # training=True penting untuk Dropout & BatchNorm
            logits = self.model(x, training=True) 
            
            # 2. Hitung Loss
            loss_value = self.loss_fn(y, logits)
            
        # 3. Hitung Gradien
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        
        # 4. Update Bobot
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss_value

    def train_epoch(self, epoch):
        self.train_loss_metric.reset_states() # Reset rata-rata loss
        
        # TQDM untuk progress bar
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch+1}", unit="batch")
        
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            # Jalankan satu step training (compiled)
            loss = self._train_step(x_batch, y_batch)
            
            # Update metric & progress bar
            self.train_loss_metric.update_state(loss)
            pbar.set_postfix({'loss': f"{self.train_loss_metric.result().numpy():.4f}"})
            
        return {'loss': self.train_loss_metric.result().numpy()}

    @tf.function
    def _valid_step(self, x, y):
        # training=False agar Dropout mati
        logits = self.model(x, training=False)
        loss_value = self.loss_fn(y, logits)
        return logits, loss_value

    def valid_epoch(self, epoch):
        self.val_loss_metric.reset_states()
        
        all_outputs = []
        all_targets = []
        
        pbar = tqdm(self.val_loader, desc=f"Valid Epoch {epoch+1}", unit="batch")
        
        for batch_idx, (x_batch, y_batch) in enumerate(pbar):
            logits, loss = self._valid_step(x_batch, y_batch)
            
            self.val_loss_metric.update_state(loss)
            pbar.set_postfix({'val_loss': f"{self.val_loss_metric.result().numpy():.4f}"})
            
            # Simpan output untuk perhitungan EER nanti
            all_outputs.append(logits) # Masih Tensor
            all_targets.append(y_batch)

        # Gabungkan semua batch jadi satu array besar (Numpy)
        # tf.concat bisa menyatukan list of tensors
        all_outputs = tf.concat(all_outputs, axis=0).numpy()
        all_targets = tf.concat(all_targets, axis=0).numpy()
        
        # TODO: Panggil metrics calculator di sini (utils/metrics.py)
        
        return {'loss': self.val_loss_metric.result().numpy()}

    def save_checkpoint(self, path):
        """
        Menyimpan bobot model TF.
        Format .h5 lebih mudah dipindah-pindah daripada SavedModel folder.
        """
        # Ubah ekstensi jadi .h5 jika user memberi .pth (karena copy paste config)
        if path.endswith('.pth'):
            path = path.replace('.pth', '.h5')
            
        try:
            self.model.save_weights(path)
            self.logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path):
        if path.endswith('.pth'):
            path = path.replace('.pth', '.h5')
            
        try:
            self.model.load_weights(path)
            self.logger.info(f"Checkpoint loaded: {path}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")