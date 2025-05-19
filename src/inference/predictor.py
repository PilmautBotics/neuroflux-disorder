import os
import datetime
from typing import List, Dict, Any

import torch
from torch import device as TorchDevice, nn
from torchvision import transforms
from torchvision.transforms import Compose
from PIL import Image
import pandas as pd
from tqdm import tqdm

from models.model_transfer import get_transfer_model

def load_model(
    model_path: str,
    model_name: str,
    num_classes: int,
    device: TorchDevice
) -> nn.Module:
    """
    Load a trained model for inference

    Arguments:
        model_path: path to the .pth weights file
        model_name: model identifier, e.g. 'transfer'
        num_classes: number of output classes
        device: torch device to use (CPU or GPU)

    Returns:
        PyTorch model ready for inference
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    if model_name == "transfer":
        model = get_transfer_model("efficientnet_b0", num_classes)
    else: 
        raise ValueError(f"Invalid model_type: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_transforms(img_size: int) -> Compose:
    """
    Define image transformations for inference

    Arguments:
        img_size: target image size (img_size x img_size)

    Returns:
        Torchvision Compose transformation pipeline
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def predict(
    model: nn.Module,
    device: TorchDevice,
    image_paths: List[str],
    transform: Compose,
    class_names: List[str]
) -> List[Dict[str, Any]]:
    """
    Run prediction on a list of images

    Arguments:
        model: trained PyTorch model
        device: torch device used
        image_paths: list of image file paths
        transform: transformation pipeline to apply
        class_names: list of class labels

    Returns:
        List of dictionaries containing predictions
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
    """
    Run the full inference pipeline using config values

    Arguments:
        config: dictionary loaded from a YAML configuration file

    Outputs:
        Saves a CSV file containing predictions
    """
    device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_names = config["general"]["class_names"]
    model_path = config["test"]["model_path"]
    img_size = int(config["dataset"]["img_size"])
    output_csv = config["test"]["output_csv"]
    input_dir = config["test"]["input_dir"]
    model_type = config["test"]["model_type"]
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not image_paths:
        raise ValueError(f"No images found in directory: {input_dir}")

    model = load_model(model_path, model_type, len(class_names), device)
    transform = get_transforms(img_size)
    results = predict(model, device, image_paths, transform, class_names)

    df = pd.DataFrame(results)

    # Create output prediction file
    model_name = os.path.basename(model_path).split('.')[0]
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    dynamic_filename = f"predictions_{model_name}_{date_str}.csv"

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, dynamic_filename)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")
