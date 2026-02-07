import os
from glob import glob

import random
import numpy as np

def collect_filepaths_and_labels(base_dir, version):
    sets = ['training', 'validation', 'testing']
    
    file_paths = { 'train': [], 'val': [], 'test': [] }
    labels = { 'train': [], 'val': [], 'test': [] }

    for s in sets:
        for cls in ['real', 'fake']:
            folder_path = os.path.join(base_dir, version, s, cls)
            # wav_files = glob(os.path.join(folder_path, '*.wav'))
            all_files = glob(os.path.join(folder_path, '*'))
            wav_files = [f for f in all_files if f.lower().endswith(('.wav', '.mp3'))]

            label = 0 if cls == 'real' else 1

            if s == 'training':
                file_paths['train'].extend(wav_files)
                labels['train'].extend([label] * len(wav_files))
            elif s == 'validation':
                file_paths['val'].extend(wav_files)
                labels['val'].extend([label] * len(wav_files))
            elif s == 'testing':
                file_paths['test'].extend(wav_files)
                labels['test'].extend([label] * len(wav_files))

    return file_paths, labels