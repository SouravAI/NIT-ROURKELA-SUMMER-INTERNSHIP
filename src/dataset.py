import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

CLASS_NAMES = ['AS', 'MR', 'MS', 'MVP', 'N']
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

class PCGDataset(Dataset):
    def __init__(self, root_dir="processed_data", transform=None):
        self.samples = []
        self.transform = transform
        
        for class_name in CLASS_NAMES:
            class_path = os.path.join(root_dir, class_name)
            files = glob(os.path.join(class_path, "*.npy"))
            for f in files:
                self.samples.append((f, CLASS_TO_INDEX[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = np.load(path).astype(np.float32)
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        # Add channel dim → [1, H, W]
        mel = torch.tensor(mel).unsqueeze(0)

        if self.transform:
            mel = self.transform(mel)

        return mel, label
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = PCGDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for batch in loader:
        x, y = batch
        print("Input shape:", x.shape)  # (B, 1, H, W)
        print("Labels:", y)
        break
