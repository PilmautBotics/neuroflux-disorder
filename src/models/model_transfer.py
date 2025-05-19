import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_transfer_model(model_name: str, num_classes: int):
    """
    Load a pretrained model
    Args:
        model_name: String name of the model to use: here is resnet18 (fixed). 
        num_classes: Number of output classes
    Returns:
        torch.nn.Module: model pretrained
    """
    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        return model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")