import matplotlib.pyplot as plt
import os
import tempfile
import mlflow
import pandas as pd 
import seaborn as sns
from sklearn.metrics import classification_report

def log_confusion_matrix(cm, class_names, step):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    with tempfile.TemporaryDirectory() as tmpdir:
        fig_path = os.path.join(tmpdir, f"confusion_matrix_epoch_{step}.png")
        plt.savefig(fig_path)
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="confusion_matrices")

def log_class_distribution_and_weights(class_counts: dict, class_names: list, step: int = 0):
    """
    Log classes distribution
    Args:
        class_counts: number of classes dict
        class_names : list of classes names 
        step : Epoch number int
    """
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    
    #_____ Display classes distribution
    classes = list(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    axes.bar(class_names, counts, color='steelblue')
    axes.set_title("Classes Distribution")
    axes.set_ylabel("Nb images")
    axes.set_xlabel("Classes")
    axes.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, f"class_dist_weights_epoch_{step}.png")
        plt.savefig(path)
        plt.close()
        mlflow.log_artifact(path, artifact_path="diagnostics")


def log_metrics_per_class(y_true: list, y_pred: list, class_names: list, step=0, prefix="val"):
    """
    Log precision, recall, F1-score for each class using MLflow.
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names of each class (same order as indices)
        step: Epoch number
        prefix: 'val' or 'train' depending on the phase
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    for cls in class_names:
        mlflow.log_metric(f"{prefix}_precision_{cls}", report[cls]["precision"], step=step)
        mlflow.log_metric(f"{prefix}_recall_{cls}", report[cls]["recall"], step=step)
        mlflow.log_metric(f"{prefix}_f1_{cls}", report[cls]["f1-score"], step=step)
        
    df = pd.DataFrame(report).T
    df = df[["precision", "recall", "f1-score", "support"]]
    print(f"\n {prefix.upper()} Classification Report at step {step}")
    print(df.to_string(float_format="%.3f"))
