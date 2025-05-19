import os
import random
from glob import glob
from sklearn.model_selection import train_test_split
from data.dataset import NeurofluxDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import re
from collections import defaultdict
import pandas as pd


import os
import torch
import torchvision.transforms.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

class ZScoreNormalize:
    "Applies Z-Score normalization followed by min-max scaling to [0, 255]."""
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image, dtype=np.float32)

        mean = image.mean()
        std = image.std() if image.std() > 0 else 1.0
        image = (image - mean) / std

        min_val, max_val = image.min(), image.max()
        if max_val - min_val < 1e-6:
            image[:] = 0
        else:
            image = (image - min_val) / (max_val - min_val) * 255

        image = image.astype(np.uint8)
        return Image.fromarray(image)

def get_transforms(img_size, augment=False):
    """Returns a composed transform pipeline for images. Multiple Data augmentation are done"""
    base = [
        ZScoreNormalize(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))
            ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)

def extract_patient_id(filename):
    """
    Extract ID of patient
    Example : 'neuroflux_002_S_1155_MR_Axial_T2-Star__...' to  '002_S_1155'
    """
    match = re.search(r'(\d{3}_S_\d{4})', filename)
    return match.group(1) if match else None


def export_dataset_csv(image_paths, labels, class_names, output_csv):
    records = []
    for path, label in zip(image_paths, labels):
        patient_id = extract_patient_id(os.path.basename(path))
        records.append({
            "image_path": path,
            "patient_id": patient_id,
            "class_idx": label,
            "class_name": class_names[label]
        })
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] CSV saved at: {output_csv}")
    
def load_data(config):
    data_dir = config["paths"]["data_dir"]
    class_names = config["general"]["class_names"]
    img_size = config["dataset"]["img_size"]
    val_split = config["dataset"]["val_split"]
    test_split = config["dataset"]["test_split"]
    augment = config["dataset"]["augmentations"]

    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    patient_to_images = defaultdict(list)
    patient_to_label = {}

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for patient_id in os.listdir(class_path):
            patient_path = os.path.join(class_path, patient_id)
            if not os.path.isdir(patient_path):
                continue

            images = glob(os.path.join(patient_path, "*"))
            if not images:
                continue

            patient_to_images[patient_id].extend(images)
            patient_to_label[patient_id] = class_to_idx[class_name]

    patient_ids = list(patient_to_images.keys())
    labels = [patient_to_label[pid] for pid in patient_ids]

    train_ids, temp_ids, y_train, y_temp = train_test_split(
        patient_ids, labels, test_size=val_split + test_split, stratify=labels, random_state=42
    )
    val_ratio = val_split / (val_split + test_split)
    val_ids, test_ids, y_val, y_test = train_test_split(
        temp_ids, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=42
    )

    def flatten(patient_ids):
        X, y = [], []
        for pid in patient_ids:
            imgs = patient_to_images[pid]
            label = patient_to_label[pid]
            X.extend(imgs)
            y.extend([label] * len(imgs))
        return X, y

    X_train, y_train = flatten(train_ids)
    X_val, y_val     = flatten(val_ids)
    X_test, y_test   = flatten(test_ids)

    train_dataset = NeurofluxDataset(X_train, y_train, transform=get_transforms(img_size, augment=augment))
    val_dataset   = NeurofluxDataset(X_val, y_val, transform=get_transforms(img_size))
    test_dataset  = NeurofluxDataset(X_test, y_test, transform=get_transforms(img_size))

    os.makedirs("debug_csv", exist_ok=True)
    export_dataset_csv(X_train, y_train, class_names, "debug_csv/train_metadata.csv")
    export_dataset_csv(X_val, y_val, class_names, "debug_csv/val_metadata.csv")
    export_dataset_csv(X_test, y_test, class_names, "debug_csv/test_metadata.csv")

    return train_dataset, val_dataset, test_dataset
