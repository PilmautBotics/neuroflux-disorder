import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Sampler
import mlflow
from tqdm import tqdm
import numpy as np
from collections import Counter
import torch.nn.functional as F
from scipy.stats import entropy as scipy_entropy
import random

from sklearn.metrics import (
    confusion_matrix,
    average_precision_score,
)

from data.prepare_dataset import load_data
from utils.visualization import *
from training.losses import FocalLoss

def get_optimizer(model, config, model_type):
    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"]["weight_decay"])

    if model_type == "transfer":
        #____ Finetuning mode
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    else:
        #____ From scratch mode, we optimize all the parameters
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)

    return optimizer


class BalancedSampler(Sampler):
    def __init__(self, labels, samples_per_class=None):
        self.labels = labels
        self.class_to_indices = self._group_indices_by_class()
        self.samples_per_class = samples_per_class or min(len(v) for v in self.class_to_indices.values())
        self.num_classes = len(self.class_to_indices)

    def _group_indices_by_class(self):
        from collections import defaultdict
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        return class_to_indices

    def __iter__(self):
        indices = []
        for label, idxs in self.class_to_indices.items():
            selected = random.choices(idxs, k=self.samples_per_class)
            indices.extend(selected)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.samples_per_class * self.num_classes

def train(config):
    device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")
    
    train_ds, val_ds, _ = load_data(config)
    
    #_____ Use Weight Random sampler to balance classes
    labels_train = train_ds.labels
    class_counts = Counter(labels_train)
    #total = sum(class_counts.values())
    #num_classes = len(class_counts)
    #class_weights = {cls: total / (num_classes * count) for cls, count in class_counts.items()}
    #weights_tensor = torch.tensor([class_weights[i] for i in range(num_classes)], dtype=torch.float32).to(device)
    #print("weights_tensor: ", weights_tensor)
    
    num_classes = int(config["general"]["num_classes"])
    model_type = config["training"]["model_type"]

    #____ Load Data and use weightedSampler
    sampler = BalancedSampler(labels_train, samples_per_class=min(Counter(labels_train).values()) * 2)
    train_loader = DataLoader(train_ds, batch_size=int(config["training"]["batch_size"]), sampler=sampler)
    
    val_loader = DataLoader(val_ds, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    
    if model_type == "transfer":
        from models.model_transfer import get_transfer_model
        print("Use transfer learning")
        model = get_transfer_model("efficientnet_b0", num_classes)
    elif model_type == "scratch":
        from models.model_scratch import MobileNetV3SmallScratch
        print("Use tranfrom scratch model")
        model = MobileNetV3SmallScratch(num_classes)
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    model = model.to(device)

    #____ define cross entropy loss for classification task
    criterion = FocalLoss(alpha=None, gamma=1.5)
    optimizer = get_optimizer(model, config, model_type)
    scheduler = None
    if config["training"].get("scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.6,
            patience=int(config["training"]["patience"]),
            min_lr= 1e-6
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
            epoch_targets = []

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                epoch_targets.extend(targets.cpu().numpy().tolist())
                    
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            #_____ Log class distribution at first epoch
            if epoch == 1: 
                class_distribution = Counter(epoch_targets)
                print("class_distribution", class_distribution)
                log_batch_distribution_image(
                class_distribution,
                config["general"]["class_names"],
                step=0
                )
                
            train_loss = total_loss / total
            train_acc = correct / total
            
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds = []
            all_labels = []
            all_entropies = []

            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    
                    #_____ get probs to compute entropy
                    probs = F.softmax(outputs, dim=1).cpu().numpy()

                    loss = criterion(outputs, targets)

                    val_loss += loss.item() * inputs.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == targets).sum().item()
                    val_total += targets.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())
                    
                    batch_entropies = scipy_entropy(probs, axis=1)
                    all_entropies.extend(batch_entropies)

            val_loss /= val_total
            val_acc = val_correct / val_total

            #___ Compute validation metrics
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            mean_entropy = np.mean(all_entropies)
            
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
            mlflow.log_metric("val_mean_entropy", mean_entropy, step=epoch)

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
