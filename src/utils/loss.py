import torch
import torch.nn as nn
from pytorch_metric_learning import losses, miners, distances

class DeepMetricLoss(nn.Module):
    """
    Unified Wrapper untuk Loss Function di Speaker Recognition / DML.
    Membungkus library 'pytorch_metric_learning' agar mudah digunakan.
    
    Support:
    1. 'triplet': TripletMarginLoss + TripletMarginMiner (Metric Learning Murni)
    2. 'arcface': ArcFaceLoss (Classification-based Metric Learning)
    3. 'contrastive': ContrastiveLoss + PairMarginMiner
    """
    def __init__(self, loss_name, embedding_size, num_classes, margin=0.2, scale=30.0):
        """
        Args:
            loss_name (str): 'arcface', 'cosface', 'triplet', 'contrastive'
            embedding_dim (int): Ukuran vektor embedding (misal 256)
            num_classes (int): Jumlah speaker (penting untuk ArcFace/CosFace)
            margin (float): Besar margin. 
                            - Untuk ArcFace (Library ini pakai Derajat): rekomen 28.6
                            - Untuk CosFace: rekomen 0.35
                            - Untuk Triplet: rekomen 0.2
            scale (float): Scaling factor (s) untuk ArcFace/CosFace (rekomen 30-64)
        """
        super().__init__()
        self.loss_name = loss_name.lower()
        
        # 1. Tentukan Distance Metric (Cosine Similarity standar untuk Audio)
        self.distance = distances.CosineSimilarity()
        
        # 2. Setup Loss Function & Miner
        self.loss_fn = None
        self.miner = None

        print(f"[Loss] Initializing {self.loss_name} | Margin={margin} | Scale={scale}")
        
        # --- GROUP 1: PAIR-BASED (Butuh Miner) ---
        if self.loss_name == 'triplet':
            # Loss Function
            self.loss_fn = losses.TripletMarginLoss(
                margin=margin, 
                distance=self.distance,
                smooth_loss=True
            )
            # Miner: Mencari pasangan 'semihard' (agak susah) agar model cepat pintar
            self.miner = miners.TripletMarginMiner(
                margin=margin, 
                distance=self.distance, 
                type_of_triplets="semihard" 
            )
            
        elif self.loss_name == 'contrastive':
            self.loss_fn = losses.ContrastiveLoss(
                pos_margin=0, 
                neg_margin=margin, 
                distance=self.distance
            )
            self.miner = miners.PairMarginMiner(
                pos_margin=0, 
                neg_margin=margin, 
                distance=self.distance
            )

        # --- GROUP 2: CLASSIFICATION-BASED (Punya Parameter 'Centers') ---
        # ArcFace/CosFace menghitung jarak sudut secara internal, jadi self.distance diabaikan di sini
        elif self.loss_name == 'arcface':
            self.loss_fn = losses.ArcFaceLoss(
                num_classes=num_classes, 
                embedding_size=embedding_size, 
                margin=margin,  # Menggunakan nilai dari Config
                scale=scale     # Menggunakan nilai dari Config
            )
            self.miner = None # ArcFace jarang butuh miner eksplisit

        elif self.loss_name == 'cosface':
            self.loss_fn = losses.CosFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=margin,
                scale=scale
            )
            self.miner = None
            
        elif self.loss_name == 'subcenter_arcface':
            self.loss_fn = losses.SubCenterArcFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=margin,
                scale=scale,
                k=3 # Jumlah sub-center per speaker
            )
            self.miner = None
            
        else:
            raise NotImplementedError(f"Loss {loss_name} belum diimplementasikan.")

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (Batch, Dim) - Output dari Model
            labels: (Batch, ) - Label Class ID
        """
        # A. Lakukan Mining (Jika ada miner)
        hard_pairs = None
        if self.miner is not None:
            hard_pairs = self.miner(embeddings, labels)
            
        # B. Hitung Loss
        # Jika hard_pairs ada, loss hanya dihitung dari pasangan tersebut
        if hard_pairs is not None:
            loss = self.loss_fn(embeddings, labels, hard_pairs)
        else:
            loss = self.loss_fn(embeddings, labels)
            
        return loss
    
    def get_trainable_params(self):
        """
        Mengembalikan parameter loss yang perlu dimasukkan ke optimizer.
        Penting untuk ArcFace/CosFace yang punya bobot 'Centers'.
        """
        if hasattr(self.loss_fn, 'parameters'):
            return list(self.loss_fn.parameters())
        return []