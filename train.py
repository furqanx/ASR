import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
import numpy as np
import random

# Import Modul Buatan Kita
from src.dataloader import get_dataloader
from src.model import build_model
from src.trainer import TorchTrainer, TFTrainer
from src.utils.loss import DeepMetricLoss

def set_seed(seed):
    """Mengatur seed agar hasil eksperimen bisa direproduksi (konsisten)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # --- 1. Load Configuration ---
    print(f"[Main] Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set Seed
    set_seed(config['experiment']['seed'])
    
    # Setup Device
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device set to: {device}")

    # --- 2. Data Pipeline ---
    print("[Main] Initializing Dataloaders...")
    # Panggil fungsi dari src/dataloader.py
    # Note: Pastikan dataloader return Raw Audio (karena model kita punya ekstraktor sendiri)
    train_loader, val_loader = get_dataloader(config['data'])

    ####################### =======================  ############################
    # print(f"Jumlah Batch di Train Loader: {len(train_loader)}")

    # # Kita ambil 1 batch saja untuk diperiksa
    # data_iter = iter(train_loader)
    # batch_audio, batch_labels = next(data_iter)

    # print("\n--- INSPEKSI KONTEN ---")
    # print(f"Shape Audio  : {batch_audio.shape}")  # Harusnya (4, 48000)
    # print(f"Shape Label  : {batch_labels.shape}")  # Harusnya (4,)
    # print(f"Contoh Label : {batch_labels}")
    # print(f"Min Value    : {batch_audio.min()}")
    # print(f"Max Value    : {batch_audio.max()}")

    # sys.exit()
    ####################### =======================  ############################

    # --- 3. Build Model ---
    print(f"[Main] Building Model: {config['model']['name']}...")
    model = build_model(config['model'])
    
    # --- 4. Setup Loss Function ---
    print(f"[Main] Setting up Loss: {config['loss']['name']}...")
    # Loss wrapper kita butuh info embedding_dim & num_classes
    loss_fn = DeepMetricLoss(
        loss_name=config['loss']['name'],
        embedding_size=config['model']['embedding_dim'],
        num_classes=config['data']['n_classes'],
        margin=config['loss']['margin']
    )
    
    # Pindahkan Loss ke Device (Penting untuk ArcFace yang punya parameter bobot)
    loss_fn = loss_fn.to(device)

    ####################### =======================  ############################
    # print("\n--- DEBUG: INSPEKSI DEEP METRIC LOSS ---")
    
    # # 1. Cek Parameter yang Bisa Dilatih (Trainable Params)
    # # ArcFace/CosFace harus punya parameter 'weight' (Centers).
    # # Triplet biasanya kosong (list kosong).
    # loss_params = list(loss_fn.parameters())
    
    # if len(loss_params) > 0:
    #     print(f"✅ Tipe Loss: Parametric (ArcFace/CosFace)")
    #     print(f"   Jumlah Parameter Tensor: {len(loss_params)}")
    #     print(f"   Shape Parameter Utama  : {loss_params[0].shape}") # Harusnya (Num_Classes, Embedding_Dim)
    #     print(f"   Lokasi Device          : {loss_params[0].device}") # HARUS sama dengan device (cuda/cpu)
    # else:
    #     print(f"ℹ️ Tipe Loss: Non-Parametric (Triplet/Contrastive)")
    
    # # 2. Simulasi Forward Pass (Uji Coba Hitung)
    # print("\n[Test] Mencoba menghitung loss dengan Dummy Data...")
    
    # try:
    #     # Buat data palsu sesuai config
    #     dummy_bs = config['data']['batch_size']
    #     dummy_dim = config['model']['embedding_dim']
    #     dummy_n_class = config['data']['n_classes']
        
    #     # Pura-pura ini output dari model (Embedding)
    #     dummy_embeddings = torch.randn(dummy_bs, dummy_dim).to(device)
    #     # Pura-pura ini label asli
    #     dummy_labels = torch.randint(0, dummy_n_class, (dummy_bs,)).to(device)
        
    #     # Hitung Loss
    #     loss_val = loss_fn(dummy_embeddings, dummy_labels)
        
    #     print(f"✅ Perhitungan Sukses!")
    #     print(f"   Input Shape : {dummy_embeddings.shape}")
    #     print(f"   Label Shape : {dummy_labels.shape}")
    #     print(f"   Loss Value  : {loss_val.item()}")
    #     print(f"   Backprop OK : {loss_val.requires_grad}") # Harus True
        
    # except Exception as e:
    #     print(f"❌ Perhitungan GAGAL!")
    #     print(f"Error Detail: {e}")
    #     # Hint debugging umum
    #     if "target" in str(e) or "index" in str(e):
    #         print("HINT: Cek apakah n_classes di config sudah sesuai dengan jumlah folder speaker?")

    # print("--- END DEBUG ---")
    # import sys; sys.exit()
    ####################### =======================  ############################

    # --- 5. Optimizer Setup ---
    print("[Main] Configuring Optimizer...")
    
    # PENTING: Gabungkan parameter Model + Parameter Loss (untuk ArcFace/CosFace)
    # Jika pakai Triplet, loss_fn.get_trainable_params() akan return list kosong []
    model_params = list(model.parameters())
    loss_params = loss_fn.get_trainable_params()
    
    all_params = model_params + loss_params
    
    lr = float(config['train']['learning_rate'])
    
    if config['train']['optimizer'] == 'adam':
        optimizer = optim.Adam(all_params, lr=lr)
    elif config['train']['optimizer'] == 'sgd':
        optimizer = optim.SGD(all_params, lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizer {config['train']['optimizer']} not supported.")

    # Scheduler (Opsional - StepLR sebagai contoh sederhana)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['train']['epochs'] // 3, gamma=0.1)

    # --- 6. Initialize Trainer ---
    backend = config['model'].get('backend', 'pytorch')
    
    if backend == 'pytorch':
        trainer = TorchTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            optimizer=optimizer,
            criterion=loss_fn, # Loss wrapper kita masuk sini
            device=device,
            scheduler=scheduler
        )
    elif backend == 'tensorflow':
        # Placeholder untuk TF
        trainer = TFTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
    else:
        raise ValueError("Backend must be 'pytorch' or 'tensorflow'")
    
    ####################### =======================  ############################
    # print("\n--- DEBUG: INSPEKSI TRAINER ---")
    
    # try:
    #     # 1. Cek Apakah Objek Terbuat
    #     print(f"✅ Trainer Instance : {type(trainer).__name__}")
        
    #     # 2. Cek Komponen Internal (Pastikan tidak None)
    #     # Kita mengakses atribut private/public trainer (tergantung implementasi TorchTrainer Anda)
    #     # Asumsi atribut disimpan dengan nama self.model, self.optimizer, dll.
        
    #     has_model = hasattr(trainer, 'model') and trainer.model is not None
    #     has_opt   = hasattr(trainer, 'optimizer') and trainer.optimizer is not None
    #     has_loss  = hasattr(trainer, 'criterion') and trainer.criterion is not None
        
    #     print(f"   - Model Linked    : {'✅' if has_model else '❌'}")
    #     print(f"   - Optimizer Linked: {'✅' if has_opt else '❌'}")
    #     print(f"   - Criterion Linked: {'✅' if has_loss else '❌'}")

    #     # 3. Cek Konsistensi Device
    #     # Pastikan model di dalam trainer berada di device yang sama dengan config
    #     trainer_device = getattr(trainer, 'device', 'Unknown')
    #     print(f"   - Trainer Device  : {trainer_device}")
        
    #     if str(trainer_device) == 'cpu' and torch.cuda.is_available():
    #         print("   ⚠️ WARNING: CUDA tersedia tapi Trainer menggunakan CPU!")

    #     # 4. Cek Ketersediaan Metode Training
    #     if hasattr(trainer, 'train_epoch') and callable(trainer.train_epoch):
    #         print(f"✅ Metode .train_epoch() ditemukan. Siap dijalankan.")
    #     else:
    #         print(f"❌ CRITICAL: Metode .train_epoch() tidak ditemukan di dalam kelas Trainer!")

    #     # 5. Cek Loader
    #     # Mencoba mengintip panjang loader dari dalam trainer
    #     if hasattr(trainer, 'train_loader'):
    #         print(f"   - Train Batches   : {len(trainer.train_loader)}")
    #     if hasattr(trainer, 'val_loader'):
    #         print(f"   - Val Batches     : {len(trainer.val_loader)}")

    # except Exception as e:
    #     print(f"❌ Gagal Menginspeksi Trainer: {e}")
    #     print("HINT: Pastikan class TorchTrainer menyimpan argumen __init__ ke dalam self.")

    # print("--- END DEBUG ---\n")
    # sys.exit() # Uncomment jika ingin berhenti disini
    ####################### =======================  ############################

    # --- 7. START TRAINING ---
    print("================================================================")
    print(f"   STARTING TRAINING FOR {config['train']['epochs']} EPOCHS")
    print("================================================================")
    
    trainer.fit(epochs=config['train']['epochs'])
    
    print("\n[Main] Training Finished Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Research Training Script")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', 
                        help='Path to the YAML config file')
    
    args = parser.parse_args()
    main(args)