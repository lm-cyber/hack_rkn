import os
from PIL import Image


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize

from transformers import ViTImageProcessor

from app.classificator import VitClassifier

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.id2label = {k: v for k, v in enumerate(sorted(os.listdir(root_dir)))}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        self.image_paths = []
        self.labels = []

        self.improcessor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        self.size = self.improcessor.size["height"]
        self.normalize = Normalize(
            mean=self.improcessor.image_mean,
            std=self.improcessor.image_std
        )

        self._transforms = Compose([
            Resize((self.size, self.size)),
            ToTensor(),
            self.normalize
        ])

        for cls in self.id2label.values():
            cls_folder = os.path.join(root_dir, cls)
            if os.path.isdir(cls_folder):
                for img_name in os.listdir(cls_folder):
                    img_path = os.path.join(cls_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(cls)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return {
            "pixel_values": self.improcessor(
                images=Image.open(self.image_paths[idx]).convert("RGB")).pixel_values[0].squeeze(), # .squeeze()
            "labels": self.label2id[self.labels[idx]]
        }
def train(output_dir="./vit_tune_results"):
    dataset = CustomImageDataset(root_dir="/home/user1/hack/train_data_rkn/dataset")
    # train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True,num_workers=4)
    
    classifier = VitClassifier(num_classes=len(set(dataset.labels))+1)
    classifier.tune(dataset,
                    device="cuda",
                    epochs=9,
                    batch_size=256,
                    lr=2e-5, test_split=False,
                    output_dir="./vit_overfit_last_results")

if __name__ == "__main__" : train()

