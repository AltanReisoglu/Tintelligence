import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import kagglehub
from PIL import Image
from torchvision import transforms
# Download latest version
path = kagglehub.dataset_download("arnaud58/landscape-pictures")
import os

IMG_SIZE = (128,128)
transform = transforms.Compose([
    
    transforms.Resize(IMG_SIZE),
   
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
gray_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),              # Boyutlandırma
    transforms.Grayscale(num_output_channels=3),  # Siyah-beyaz (tek kanal)
    transforms.ToTensor(),                   # [0, 255] → [0.0, 1.0]
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))     # Tek kanal için normalize
])

class FlatImageDataset(Dataset):
    def __init__(self, folder_path, transform=transform,gray_t=gray_transform):
        self.image_paths = [os.path.join(folder_path, fname)
                            for fname in os.listdir(folder_path)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.gray_transform=gray_t

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            gray_img=self.gray_transform(img)
            target_img = self.transform(img)
        return gray_img,target_img

dataset=FlatImageDataset(folder_path=path,transform=transform,gray_t=gray_transform)

train_dataloaded=DataLoader(dataset,1,shuffle=True,drop_last=True)

if __name__=="__main__":
    for k,j in train_dataloaded:
        print(k.shape)
        print(j.shape)
        break
        