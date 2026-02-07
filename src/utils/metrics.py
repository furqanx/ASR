import numpy as np
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_fscore_support
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_accuracy(outputs, labels):
    """
    Menghitung akurasi standar untuk tugas Klasifikasi (Supervised).
    
    Args:
        outputs (np.array): Logits/Softmax output dari model (Batch, N_Classes).
        labels (np.array): Label asli (Batch, ).
    
    Returns:
        float: Akurasi (0.0 - 1.0)
    """
    # Ambil index dengan nilai tertinggi (argmax)
    if outputs.ndim > 1:
        preds = np.argmax(outputs, axis=1)
    else:
        preds = outputs
        
    return accuracy_score(labels, preds)

def compute_eer(embeddings, labels, metric='cosine'):
    """
    Menghitung Equal Error Rate (EER) untuk tugas Verifikasi (DML).
    Menggunakan protokol 'All-vs-All': Membandingkan setiap embedding dengan 
    seluruh embedding lainnya dalam batch validasi.
    
    Args:
        embeddings (np.array): Vektor fitur (Batch, Dimensi).
        labels (np.array): ID Pembicara/Kelas (Batch, ).
        metric (str): 'cosine' atau 'euclidean'.
    
    Returns:
        eer (float): Nilai Error (0.0 - 1.0). Semakin kecil semakin bagus.
        threshold (float): Ambang batas optimal.
    """
    # 1. Normalisasi Embedding (Penting untuk Cosine Similarity)
    # Agar cosine similarity sama dengan dot product
    if metric == 'cosine':
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10) # Tambah epsilon agar tidak div by zero

    # 2. Hitung Similarity Matrix (N x N)
    # Dot product: (N, D) x (D, N) -> (N, N)
    sim_matrix = np.matmul(embeddings, embeddings.T)
    
    # 3. Buat Label Matrix (Ground Truth)
    # mask[i][j] = 1 jika label[i] == label[j] (Pasangan Positif)
    # mask[i][j] = 0 jika label[i] != label[j] (Pasangan Negatif)
    labels = labels.reshape(-1, 1)
    label_matrix = (labels == labels.T).astype(int)
    
    # 4. Ambil Segitiga Atas (Upper Triangle)
    # Kita tidak butuh diagonal (jarak ke diri sendiri)
    # Kita tidak butuh segitiga bawah (duplikasi segitiga atas)
    tri_u_indices = np.triu_indices(sim_matrix.shape[0], k=1)
    
    scores = sim_matrix[tri_u_indices]
    truth = label_matrix[tri_u_indices]
    
    # 5. Hitung ROC Curve menggunakan Scikit-Learn
    # fpr: False Positive Rate
    # tpr: True Positive Rate
    fpr, tpr, thresholds = roc_curve(truth, scores, pos_label=1)
    
    # 6. Hitung EER (Equal Error Rate)
    # EER adalah titik dimana False Acceptance Rate (FPR) == False Rejection Rate (1 - TPR)
    # fnr = 1 - tpr
    # Kita cari nilai x dimana fpr(x) - fnr(x) = 0
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # Cari threshold yang menghasilkan EER tersebut
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer, thresh

def compute_metrics_report(outputs, labels, mode='classification'):
    """
    Fungsi Helper Utama yang dipanggil oleh Trainer.
    Otomatis memilih metric berdasarkan mode.
    """
    results = {}
    
    if mode == 'classification':
        # Untuk Supervised Learning biasa
        acc = compute_accuracy(outputs, labels)
        results['accuracy'] = acc
        
        # Tambahan: Precision, Recall, F1 (Macro average)
        if outputs.ndim > 1:
            preds = np.argmax(outputs, axis=1)
        else:
            preds = outputs
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
        results['precision'] = p
        results['recall'] = r
        results['f1'] = f1
        
    elif mode == 'dml' or mode == 'metric_learning':
        # Untuk Speaker Recognition / DML
        eer, thresh = compute_eer(outputs, labels)
        results['eer'] = eer
        results['threshold'] = thresh
        
    return results