import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
import numpy as np
from collections import Counter

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

from data.prepare_dataset import load_data
from utils.visualization import log_class_distribution_and_weights, log_confusion_matrix, log_metrics_per_class

def get_optimizer(model, config, model_type):
    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"]["weight_decay"])

    if model_type == "transfer":
        #____ Finetuning mode
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        #____ From scratch mode, we optimize all the parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)

    return optimizer

def train(config):
    device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")
    
    train_ds, val_ds, _ = load_data(config)
    
    #_____ Use Weight Random sampler to balance classes
    labels_train = [label for _, label in train_ds]
    class_counts = Counter(labels_train)

    train_loader = DataLoader(train_ds, batch_size=int(config["training"]["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config["training"]["batch_size"]), shuffle=False)
     
    num_classes = int(config["general"]["num_classes"])
    model_type = config["training"]["model_type"]

    # ____ Compute class weights to not over learn on majority class EO.
    counts = np.bincount(labels_train, minlength=num_classes)
    total = counts.sum()
    weights = total / (num_classes * counts + 1e-6)
    weights = (weights / weights.sum()) ** 1.5
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    if model_type == "transfer":
        from models.model_transfer import get_transfer_model
        model = get_transfer_model("efficientnet_b0", num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    model = model.to(device)

    #____ define cross entropy loss for classification task
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  
    
    optimizer = get_optimizer(model, config, model_type)
    scheduler = None
    if config["training"].get("scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.2,
            patience=int(config["training"]["patience"]),
            min_lr= 1e-6,
            verbose=True
        )

    mlflow.set_tracking_uri("file:///" + os.path.abspath(config["paths"]["log_dir"]))
    mlflow.set_experiment("neuroflux_classification")

    with mlflow.start_run():
        mlflow.log_params(config["training"])
        # ____ log class distribution and weights associated
        log_class_distribution_and_weights(
            class_counts=class_counts,
            class_names=config["general"]["class_names"],
            step=0
        )

        best_val_loss = float("inf")
        patience = int(config["training"]["patience"])
        use_early_stopping = config["training"].get("early_stopping", False)
        counter = 0

        for epoch in range(1, config["training"]["epochs"] + 1):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            train_loss = total_loss / total
            train_acc = correct / total

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())

            val_loss /= val_total
            val_acc = val_correct / val_total

            #___ Compute validation metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            log_metrics_per_class(y_true=all_labels, y_pred=all_preds, class_names=config["general"]["class_names"], step=epoch, prefix="val")

            try:
                mAP = average_precision_score(
                    np.eye(config["general"]["num_classes"])[all_labels],
                    np.eye(config["general"]["num_classes"])[all_preds],
                    average="macro"
                )
            except:
                mAP = 0.0

            #_____ Train logging metrics to MLFlow monitoring 
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            #_____ Val logging metrics to MLFlow monitoring
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("val_mAP", mAP, step=epoch)

            #_____ Compute confusion matrix and log at each epoch
            cm = confusion_matrix(all_labels, all_preds)
            log_confusion_matrix(cm, config["general"]["class_names"], epoch)
            print("Confusion matrix:\n", cm)

            print(f"Epoch {epoch}/{config['training']['epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(config["paths"]["model_save_path"]), exist_ok=True)
                torch.save(model.state_dict(), config["paths"]["model_save_path"])
                counter = 0
                print("Model checkpoint saved.")
            else:
                counter += 1
                if use_early_stopping and counter >= patience:
                    print("Early stopping.")
                    break

            if scheduler:
                scheduler.step(val_loss)
