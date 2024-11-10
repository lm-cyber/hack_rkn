from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision as tv
import tqdm
import onnxruntime as ort
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import AveragePrecision
from transformers import (
    DefaultDataCollator,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)


import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
# import numpy as np


class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            print(
                f"Epoch {state.epoch} - Step {state.global_step} - Loss: {logs['loss']}"
            )


class AnyClassifier:
    def __call__(self, *args, **kwargs):
        pass


class VitClassifier(AnyClassifier):
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        id2label=None,
        label2id=None,
    ):

        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            # num_labels=num_classes,
            attn_implementation="sdpa",
            torch_dtype=torch.float32,
            ignore_mismatched_sizes=True,
            id2label=id2label,
            label2id=label2id,
        )

        self.image_processor = ViTImageProcessor.from_pretrained(model_name)

    def __call__(self, image: Image.Image, device: str = "cpu"):
        image = (
            self.improcessor(images=Image.open(self.image_paths[idx]).convert("RGB"))
            .pixel_values[0]
            .squeeze()
        )

        self.model.to(device)
        with torch.no_grad():
            outputs = self.model(image).logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()

        return predicted_class, probabilities.squeeze().cpu().numpy()

    def transform(self, image: Image.Image):
        return self.image_processor(image)["pixel_values"].squeeze(
            0
        )

    def tune(
        self,
        data: Dataset,
        epochs: int = 3,
        device: str = "cpu",
        batch_size: int = 32,
        lr: float = 1e-6,
        output_dir: str = "./vit_results",
        test_split=True,
    ):

        if test_split:
            train_size = int(0.8 * len(data))
            test_size = len(data) - train_size
            train_data, test_data = random_split(data, [train_size, test_size])
        else:
            train_data = test_data = data

        def compute_metrics(eval_pred):
            preds = torch.tensor(eval_pred[0])
            labels = torch.tensor(eval_pred[1])

            average_precision = AveragePrecision(
                task="multiclass", num_classes=self.model.config.num_labels
            )
            return {"mean_average_precision": average_precision(preds, labels).item()}

        def lr_lambda(current_epoch):
            return 0.1 ** (current_epoch // lr_decay_epoch)

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        # lr_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch % lr_decay_epoch == 0 else 1.0)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            compute_metrics=compute_metrics,
            data_collator=DefaultDataCollator(),
            # optimizers=(optimizer, lr_scheduler),
            callbacks=[LossLoggerCallback()],
        )

        trainer.train()
        print("Fine-tuning complete.")


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

    def tune(
        self,
        data: Dataset,
        epochs: int = 3,
        device: str = "cpu",
        batch_size: int = 64,
        dl_num_workers: int = 4,
        lr=1e-6,
    ):

        train_size = int(0.8 * len(data))
        test_size = len(data) - train_size
        train_data, test_data = random_split(data, [train_size, test_size])

        train_loader = DataLoader(
            train_data, num_workers=dl_num_workers, batch_size=batch_size, shuffle=True
        )

        test_loader = DataLoader(
            test_data, num_workers=dl_num_workers, batch_size=batch_size
        )

        self.backbone.to(device)
        self.head.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.head.parameters()), lr=0.0003
        )
        ap_metric = AveragePrecision(
            task="multiclass", num_classes=self.head.out_features
        ).to(device)

        for ep in range(epochs):
            running_loss = 0.0
            self.backbone.train()
            self.head.train()

            with tqdm.tqdm(
                train_loader, total=len(train_loader), desc=f"Epoch {ep+1}/{epochs}"
            ) as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    features = self.backbone(inputs)
                    outputs = self.head(features)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix(loss=running_loss / len(train_loader))

            self.backbone.eval()
            self.head.eval()
            with torch.no_grad():
                ap_metric.reset()
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    features = self.backbone(inputs)
                    outputs = self.head(features)
                    ap_metric.update(outputs, labels)

            mAP = ap_metric.compute().item()
            print(f"Epoch {ep+1}/{epochs} - Mean Average Precision: {mAP:.4f}")

        print("Fine-tuning complete.")



class ClassificatorONNX:
    def __init__(self, model_path: str, device = "cpu"):
        self.session = ort.InferenceSession(model_path, providers = ["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"])

        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize((224, 224)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _transform(self, image: Image.Image) -> np.ndarray:
        processed_image = self.image_processor(image).unsqueeze(0)
        return processed_image.numpy().astype(np.float32)

    def predict_proba_class(self, image: Image.Image) -> List[np.ndarray]:
        processed_image = self._transform(image)
    
        outputs = self.session.run(None, {"pixel_values": processed_image})
        probabilities = (np.exp(outputs)/np.sum(np.exp(outputs)))[0][0]
        predicted_class = probabilities.argmax()
       
        return predicted_class, probabilities[predicted_class]

    def predict(self, image: Image.Image) -> int:
        return self.predict_proba_class(image)[0]

    def predict_embedding(self, image: Image.Image) -> np.ndarray:
        processed_image = self._transform(image)
        outputs = self.session.run(None, {"pixel_values": processed_image})
        embedding = outputs[0][0]
        return embedding.squeeze()

    def predict_result(self, image: Image.Image) -> Dict[str, np.ndarray]:
        predicted_class, probabilities = self.predict_proba_class(image)
        embedding = self.predict_embedding(image)

        return {
            "class": predicted_class,
            "probs_class": probabilities,
            "embedding": embedding,
        }