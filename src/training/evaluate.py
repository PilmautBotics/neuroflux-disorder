import os
import torch
import numpy as np
from tqdm import tqdm
import mlflow

from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
)

from data.prepare_dataset import load_data
from models.model_transfer import get_transfer_model
from utils.visualization import log_confusion_matrix, log_metrics_per_class


def evaluate(config):
    device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")

    #_________ Load datasets test are needed because validation is already compute at training
    _, _, test_ds = load_data(config)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False
    )

    num_classes = int(config["general"]["num_classes"])
    class_names = config["general"]["class_names"]

    #_________ Load model
    model_type = config["training"]["model_type"]
    if model_type == "transfer":
        model = get_transfer_model("efficientnet_b0", num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    model.load_state_dict(torch.load(config["paths"]["model_save_path"], map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0
    total = 0

    #_________ Loss function with optional weighting
    weights = config["training"].get("class_weights", None)
    if weights:
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    
    if not config["paths"].get("log_dir"):
        raise ValueError(f"No path for outputs results: {config['paths'].get('log_dir')}")
    
    mlflow.set_tracking_uri("file:///" + os.path.abspath(config["paths"]["log_dir"]))
    mlflow.set_experiment("neuroflux_classification_test")
    with mlflow.start_run(run_name="evaluation"):

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluating"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())
                total += targets.size(0)

        avg_loss = total_loss / total
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        #_________ Compute metrics
        log_metrics_per_class(y_true=all_labels, y_pred=all_preds, class_names=config["general"]["class_names"], step=0, prefix="test")
        
        try:
            mAP = average_precision_score(
                np.eye(num_classes)[all_labels],
                np.eye(num_classes)[all_preds],
                average="macro"
            )
        except:
            mAP = 0.0

        cm = confusion_matrix(all_labels, all_preds)
        print("\nEvaluation Results:")
        print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | mAP: {mAP:.4f}")
        print("Confusion Matrix:\n", cm)
        
        mlflow.log_metric("eval_loss", avg_loss)
        mlflow.log_metric("eval_accuracy", accuracy)
        mlflow.log_metric("eval_mAP", mAP)
        log_confusion_matrix(cm, class_names, step=0)
        
        print("You can now explore the results with mlflow under neuroflux_classification_test")

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config)
