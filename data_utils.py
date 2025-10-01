import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.utils as vutils
from PIL import Image
import math

# Define the dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, label_prefix, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, file) 
            for file in os.listdir(root_dir) 
            if file.startswith(label_prefix + "_") and file.endswith(".jpg")
        ]
        self.label_prefix = label_prefix

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel images
        if self.transform:
            image = self.transform(image)
        return image, self.label_prefix  

# Function from https://github.com/SilverEngineered/MosaiQ
def scale_data(data, scale=None, dtype=np.float32):
    if scale is None:
        scale = [-1, 1]
    min_data, max_data = [float(np.min(data)), float(np.max(data))]
    min_scale, max_scale = [float(scale[0]), float(scale[1])]
    data = ((max_scale - min_scale) * (data - min_data) / (max_data - min_data)) + min_scale
    return data.astype(dtype)

# Function from https://github.com/SilverEngineered/MosaiQ
def get_noise_upper_bound(gen_loss, disc_loss, original_ratio):
    R = disc_loss.detach().cpu().numpy()/gen_loss.detach().cpu().numpy()

    return math.pi/8 + (5 *math.pi / 8) * relu(np.tanh((R - (original_ratio))))


