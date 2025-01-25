from torch.utils.data import Dataset
import os
import pickle
import pandas as pd
import numpy

class ApkEmbDataset(Dataset):
    def __init__(self, root_dir, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.emb_dir = root_dir
        self.hash_list = []

        for _, row in self.data_frame.iterrows():
            sha256 = row['sha256']
            label = 1 if row['malware'] else 0
            emb_path = os.path.join(root_dir, 'malware' if label else 'goodware', sha256 + '.pkl')

            # Check if the file exists before adding to the list
            if os.path.exists(emb_path):
                self.hash_list.append([emb_path, label])
            else:
                print(f"File not found: {emb_path}")

    def __len__(self):
        return len(self.hash_list)

    def __getitem__(self, index):
        emb_path, label = self.hash_list[index]
        try:
            apk_emb = pickle.load(open(emb_path, 'rb'))
            if not isinstance(apk_emb, numpy.ndarray):
                raise ValueError(f"Unexpected type for {emb_path}: {type(apk_emb)}")
        except Exception as e:
            print(f"Error loading {emb_path}: {e}")
            raise RuntimeError(f"Failed to load embedding for index {index}")
        
        return apk_emb, label
