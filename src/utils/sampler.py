import os
from glob import glob

import random
import numpy as np
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Sampler



import numpy as np
import random
from collections import defaultdict

class BalancedBatchScheduler:
    """
    Otak pengatur antrian.
    Tugasnya hanya menghasilkan urutan INDEX (misal: [0, 5, 12, 3...])
    yang menjamin P speaker x K utterance.
    """
    def __init__(self, labels, n_classes, n_samples):
        self.labels = np.array(labels)
        self.n_classes = n_classes # P (Jumlah orang per batch)
        self.n_samples = n_samples # K (Jumlah suara per orang)
        self.batch_size = self.n_classes * self.n_samples
        
        # 1. Kelompokkan Index berdasarkan Label Speaker
        # {0: [1, 5, 9], 1: [2, 6, 10], ...}
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
            
        self.used_label_indices_count = {label: 0 for label in self.label_to_indices}
        self.count = 0
        self.n_dataset = len(self.labels)

    def __iter__(self):
        """
        Menghasilkan generator index
        """
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # A. Pilih P speaker secara acak
            classes = np.random.choice(list(self.label_to_indices.keys()), self.n_classes, replace=False)
            
            indices = []
            for class_ in classes:
                # B. Ambil K sampel dari speaker tersebut
                available_indices = self.label_to_indices[class_]
                
                # Strategi: Random sampling with replacement (agar tidak habis)
                # atau Linear sampling (agar semua kena). 
                # Disini kita pakai Random Choice agar simpel & robust.
                selected_indices = np.random.choice(available_indices, self.n_samples, replace=True)
                indices.extend(selected_indices)
                
            yield indices # Return 1 Batch Index (P x K)
            self.count += self.batch_size

    def __len__(self):
        return self.n_dataset // self.batch_size
    



class TorchBalancedSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.scheduler = BalancedBatchScheduler(labels, n_classes, n_samples)
        
    def __iter__(self):
        # Flatten list of lists menjadi single stream of indices
        for batch_indices in self.scheduler:
            for idx in batch_indices:
                yield idx

    def __len__(self):
        return self.scheduler.__len__() * self.scheduler.batch_size
    

# class BalancedSpeakerGenerator:
#     def __init__(self, 
#           file_paths, labels, 
#           n_per_speaker, 
#           max_seg_per_spk
#         ):
#         self.file_paths = file_paths
#         self.labels = labels
#         self.n_per_speaker = n_per_speaker
#         self.max_seg_per_spk = max_seg_per_spk

#         # 1. Kelompokkan data berdasarkan Speaker ID
#         # Format: { label_id: [index_file_1, index_file_2, ...] }
#         self.data_dict = {}
#         for idx, label in enumerate(labels):
#             if label not in self.data_dict:
#                 self.data_dict[label] = []
#             self.data_dict[label].append(idx)

#         self.speaker_list = list(self.data_dict.keys())

#     def __call__(self):
#         """
#         Ini adalah fungsi generator yang akan dipanggil oleh tf.data
#         """
#         # A. Acak urutan speaker (agar batch tidak isinya orang itu-itu saja tiap epoch)
#         random.shuffle(self.speaker_list)

#         # B. Loop per speaker
#         for speaker_label in self.speaker_list:
#             indices = self.data_dict[speaker_label]
            
#             # Ambil sampel secara acak dari speaker ini (maksimal max_seg_per_spk)
#             # Logika PyTorch: round_down(min(len, max_seg), nPerSpeaker)
#             # Kita sederhanakan: Ambil chunk yang habis dibagi n_per_speaker
            
#             # 1. Shuffle indeks file milik speaker ini
#             random.shuffle(indices)
            
#             # 2. Tentukan berapa banyak sampel yang diambil
#             # Harus kelipatan n_per_speaker (misal 2, 4, 6...)
#             num_samples = min(len(indices), self.max_seg_per_spk)
#             num_samples = num_samples - (num_samples % self.n_per_speaker)

#             if num_samples < self.n_per_speaker:
#                 continue # Skip jika sampel kurang dari syarat minimal

#             selected_indices = indices[:num_samples]

#             # C. Yield data satu per satu (Streaming)
#             # TF akan menangkap ini dan nanti menyatukannya jadi Batch
#             for idx in selected_indices:
#                 yield self.file_paths[idx], self.labels[idx]
