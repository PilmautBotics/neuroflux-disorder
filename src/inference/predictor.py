"""Inference module for the Neuroflux classification project.

This module provides functionality for running inference on new images using a
trained model. It includes functions for loading models, preprocessing images,
and generating predictions with confidence scores.
"""

import os
import datetime
from typing import List, Dict, Any

import pandas as pd
import torch
from PIL import Image
from torch import device as TorchDevice, nn
from torchvision import transforms
from torchvision.transforms import Compose
from tqdm import tqdm

from models.model_transfer import get_transfer_model
from models.model_scratch import MobileNetV3SmallScratch


def load_model(
    model_path: str,
    model_name: str,
    num_classes: int,
    device: TorchDevice
) -> nn.Module:
    """Load a trained model for inference.

    Args:
        model_path (str): Path to the .pth weights file.
        model_name (str): Model identifier, e.g. 'transfer' or 'scratch'.
        num_classes (int): Number of output classes.
        device (TorchDevice): Torch device to use (CPU or GPU).

    Returns:
        nn.Module: PyTorch model ready for inference.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If model_name is not recognized.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if model_name == "transfer":
        model = get_transfer_model("efficientnet_b0", num_classes)
    elif model_name == "scratch":
        model = MobileNetV3SmallScratch(num_classes)
    else: 
        raise ValueError(f"Invalid model_type: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transforms(img_size: int) -> Compose:
    """Define image transformations for inference.

    The transformation pipeline includes resizing, conversion to tensor,
    and normalization using ImageNet statistics.

    Args:
        img_size (int): Target image size (img_size x img_size).

    Returns:
        Compose: Torchvision transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def predict(
    model: nn.Module,
    device: TorchDevice,
    image_paths: List[str],
    transform: Compose,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """Run prediction on a list of images.

    Args:
        model (nn.Module): Trained PyTorch model.
        device (TorchDevice): Torch device to use.
        image_paths (List[str]): List of image file paths.
        transform (Compose): Transformation pipeline to apply.
        class_names (List[str]): List of class labels.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing predictions
            with keys 'image', 'predicted_class', and 'confidence'.
    """
    results = []

    for img_path in tqdm(image_paths, desc="Predicting", unit="image"):
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue

        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = class_names[probabilities.argmax().item()]
            confidence = probabilities.max().item()

        results.append({
            'image': os.path.basename(img_path),
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    return results


def inference(config: Dict[str, Any]) -> None:
    """Run the full inference pipeline using config values.

    This function:
    1. Sets up the device and loads configuration
    2. Loads the trained model
    3. Processes all images in the input directory
    4. Saves predictions to a CSV file

    Args:
        config (Dict[str, Any]): Dictionary loaded from a YAML configuration file.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        ValueError: If no images are found in input_dir.
    """
    device = torch.device(
        config["general"]["device"] if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load configuration values
    class_names = config["general"]["class_names"]
    model_path = config["test"]["model_path"]
    img_size = int(config["dataset"]["img_size"])
    output_csv = config["test"]["output_csv"]
    input_dir = config["test"]["input_dir"]
    model_type = config["test"]["model_type"]
    
    # Validate input directory
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Get list of image files
    image_paths = [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_paths:
        raise ValueError(f"No images found in directory: {input_dir}")

    # Run inference
    model = load_model(model_path, model_type, len(class_names), device)
    transform = get_transforms(img_size)
    results = predict(model, device, image_paths, transform, class_names)

    # Save results
    df = pd.DataFrame(results)
    model_name = os.path.basename(model_path).split('.')[0]
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    dynamic_filename = f"predictions_{model_name}_{date_str}.csv"

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, dynamic_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
