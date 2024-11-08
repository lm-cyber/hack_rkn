

from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision as tv
from torchvision.models import resnet34, ResNet

import tqdm
from PIL import Image
# import numpy as np 

class AnyClassifier:
    def __call__(self, *args, **kwargs):
        pass

    def batch_classifier(self, *args, **kwargs):
        pass


class ResClassifier(AnyClassifier):
    def __init__(self, model_installer: callable, num_classes: int = 500):
        self.backbone: ResNet = model_installer()
        self.backbone_block_expansion = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(self.backbone_block_expansion, num_classes)

        self.transform = tv.transforms.Compose(
            [
                tv.transforms.Resize((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image: Image.Image, device: str = "cpu"):
        image = self.transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.head(self.backbone(image))
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()

        return predicted_class, probabilities.squeeze().cpu().numpy()

    def batch_classificator(self, images: List[Image.Image], device="cpu"):
        processed_images = [self.model.transform(img).unsqueeze(0) for img in images]

        batch = torch.cat(processed_images, dim=0).to(device)
        with torch.no_grad():
            logits = self.model.head(self.model.backbone(batch))
            probabilities = torch.softmax(logits, dim=1)
            predicted_classes = probabilities.argmax(dim=1).cpu().numpy()

        return predicted_classes, probabilities.cpu().numpy()

    def tune(self, data: DataLoader, epochs: int = 3, device: str = "cpu"):
        self.backbone.to(device)
        self.head.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.head.parameters())
        )

        for ep in range(epochs):
            running_loss = 0.0
            with tqdm(data, total=len(data), desc=f"Epoch {ep+1}/{epochs}") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    features = self.backbone(inputs)
                    outputs = self.head(features)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix(loss=running_loss / len(data))

        print("Fine-tuning complete.")


class Classificator:
    def __init__(self, model: AnyClassifier):
        self.model = model()

    def __call__(self, image: Image.Image, device="cpu"):
        return self.model.__call__(image, device=device)

    def batch_classifier(self, images: List[Image.Image], device="cpu"):
        return self.model.batch_classifier(images, device=device)


classificator_instance = Classificator()
