"""Visualization utilities for model training and evaluation.

This module provides functions for visualizing and logging various metrics and
diagnostics during model training and evaluation, including confusion matrices,
class distributions, and per-class metrics.
"""

import os
import tempfile
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report


def log_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    step: int
) -> None:
    """Log a confusion matrix visualization to MLflow.

    Args:
        cm (List[List[int]]): Confusion matrix as a 2D list of integers.
        class_names (List[str]): List of class names for axis labels.
        step (int): Current step/epoch number.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, f"confusion_matrix_epoch_{step}.png")
        plt.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="confusion_matrices")


def log_batch_distribution_image(
    counts: Dict[int, int],
    class_names: List[str],
    step: int = 0
) -> None:
    """Log a bar plot of class distribution in the current batch.

    Args:
        counts (Dict[int, int]): Dictionary mapping class indices to counts.
        class_names (List[str]): List of class names for x-axis labels.
        step (int, optional): Current epoch number. Defaults to 0.
    """
    values = [counts.get(i, 0) for i in range(len(class_names))]

    fig, ax = plt.subplots()
    ax.bar(class_names, values, color="skyblue")
    ax.set_title(f"Batch Class Distribution (Epoch {step})")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.tick_params(axis='x', rotation=45)

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, f"batch_distribution_epoch_{step}.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="diagnostics")


def log_class_distribution_and_weights(
    class_counts: Dict[int, int],
    class_names: List[str],
    step: int = 0
) -> None:
    """Log overall class distribution visualization.

    Args:
        class_counts (Dict[int, int]): Dictionary mapping class indices to counts.
        class_names (List[str]): List of class names for x-axis labels.
        step (int, optional): Current epoch number. Defaults to 0.
    """
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    
    # Display classes distribution
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    axes.bar(class_names, counts, color='steelblue')
    axes.set_title("Classes Distribution")
    axes.set_ylabel("Number of Images")
    axes.set_xlabel("Classes")
    axes.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"class_dist_weights_epoch_{step}.png")
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="diagnostics")


def log_metrics_per_class(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    step: int = 0,
    prefix: str = "val"
) -> None:
    """Log precision, recall, and F1-score for each class using MLflow.

    Args:
        y_true (List[int]): Ground truth labels.
        y_pred (List[int]): Predicted labels.
        class_names (List[str]): Names of each class (same order as indices).
        step (int, optional): Current epoch number. Defaults to 0.
        prefix (str, optional): Phase identifier ('val' or 'train').
            Defaults to "val".
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    for cls in class_names:
        mlflow.log_metric(
            f"{prefix}_precision_{cls}",
            report[cls]["precision"],
            step=step
        )
        mlflow.log_metric(
            f"{prefix}_recall_{cls}",
            report[cls]["recall"],
            step=step
        )
        mlflow.log_metric(
            f"{prefix}_f1_{cls}",
            report[cls]["f1-score"],
            step=step
        )
        
    df = pd.DataFrame(report).T
    df = df[["precision", "recall", "f1-score", "support"]]
    print(f"\n{prefix.upper()} Classification Report at step {step}")
    print(df.to_string(float_format="%.3f"))
