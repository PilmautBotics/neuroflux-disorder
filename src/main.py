"""Entry point for training, evaluation, and prediction of the Neuroflux model.

This script allows launching one of three modes:
    - train: trains a classification model on medical images
    - evaluate: evaluates a trained model on a validation/test set
    - predict: performs inference on new unseen data
"""

import argparse
import yaml


def load_config(config_path):
    """Load configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_model(config):
    """Train a new model using the provided configuration.

    Args:
        config (dict): Configuration dictionary.
    """
    from training.trainer import train
    train(config)


def evaluate_model(config):
    """Evaluate a trained model using the provided configuration.

    Args:
        config (dict): Configuration dictionary.
    """
    from training.evaluate import evaluate
    evaluate(config)


def predict_image(config):
    """Run inference on new images using a trained model.

    Args:
        config (dict): Configuration dictionary.
    """
    from inference.predictor import inference
    inference(config)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Neuroflux Disorder Classifier")
    parser.add_argument(
        "mode",
        choices=["train", "evaluate", "predict", "test"],
        help="Mode to run the script"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "train":
        train_model(config)
    elif args.mode == "evaluate":
        evaluate_model(config)
    elif args.mode == "predict":
        predict_image(config)
    elif args.mode == "test":
        print("Main parser works fine!")
    else:
        raise ValueError(f"Unknown mode {args.mode}")


if __name__ == "__main__":
    main()
