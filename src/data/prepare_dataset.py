import os
import random
from glob import glob
from sklearn.model_selection import train_test_split
from data.dataset import NeurofluxDataset
from torchvision import transforms
from PIL import Image
import numpy as np

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
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]

    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)

def load_data(config):
    """
    Loads and splits dataset into train, validation and test sets.
    Args:
        config: yaml config dict with parameters.
    Returns:
        Tuple of train, val, test sets
    """
    data_dir = config["paths"]["data_dir"]
    class_names = config["general"]["class_names"]
    img_size = config["dataset"]["img_size"]
    val_split = config["dataset"]["val_split"]
    test_split = config["dataset"]["test_split"]
    augment = config["dataset"]["augmentations"]

    all_image_paths = []
    all_labels = []

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    for cls_name in class_names:
        folder = os.path.join(data_dir, cls_name)
        images = glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png"))

        all_image_paths += images
        all_labels += [class_to_idx[cls_name]] * len(images)

    combined = list(zip(all_image_paths, all_labels))
    random.shuffle(combined)
    all_image_paths, all_labels = zip(*combined)

    #______ Split into train, val, test
    X_temp, X_test, y_temp, y_test = train_test_split(all_image_paths, all_labels, test_size=test_split, stratify=all_labels)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split / (1 - test_split), stratify=y_temp)

    train_dataset = NeurofluxDataset(X_train, y_train, transform=get_transforms(img_size, augment=augment))
    val_dataset   = NeurofluxDataset(X_val, y_val, transform=get_transforms(img_size))
    test_dataset  = NeurofluxDataset(X_test, y_test, transform=get_transforms(img_size))

    return train_dataset, val_dataset, test_dataset
