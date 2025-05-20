# Neuroflux Disorder Classifier

A Deep Learning project to classify 2D MRI images into 5 phases of a fictional disease, *Neuroflux Disorder*:
- EO: Early Onset
- IO: Intermediate Onset
- LO: Late Onset
- PTE: Polyglutamine Tract Expansion
- IPTE: Intermediate Polyglutamine Tract Expansion

The goal is to develop and compare two models:
1. `efficientnet_b0`: Transfer learning using a pretrained CNN.
2. `mobilenetV3-small`: MobileNetV3-small implemented from scratch.


## Project Structure

```
.
├── configs/                # YAML config files 
├── data/                   # Dataset loaders and pre-processing
├── models/                 # Model definitions (transfer and scratch)
├── notebooks/              # Exploration and visual inspection for dataset given
├── src/
│   ├── training/           # Training loop and evaluation scripts
│   ├── inference/          # Prediction script
│   ├── utils/              # Logging, metrics, visualizations
│   └── main.py             # Entry point for training / inference / evaluation
├── Dockerfile              # Docker image with GPU support
└── README.md               # Project documentation
```

## Setup

### Requirements
Please install docker on your machine via : https://docs.docker.com/get-started/

Install dependencies in a virtual environment/ Conda or Minconda:

```bash
pip install -r requirements.txt
```

Or build the Docker image:

```bash
docker build -t neuroflux-classifier .
```

Or pull the Docker image via public docker Hub repo: 

```bash
docker pull pilmaut/neuroflux-classifier:latest
```

NB: if you use conda installation, please make sure you have the right version install for cuda, cudnn and pytorch
---

## Getting Started

 -- TODO

## Usage 

### 1. Train the model

```bash
python main.py train -c ./configs/config_debug.yaml
```

### 2. Evaluate a trained model

```bash
python main.py evaluate -c ./configs/config_debug.yaml
```

### 3. Predict from new images

```bash
python main.py predict -c ./configs/config_debug.yaml
```

## Monitoring with MLflow

To launch MLflow UI on explore the results generated you can use the command below:
 MLFLOW: mlflow ui --backend-store-uri file:./src/logs_debug/ --port 5050

## Example Config (YAML) for training / evaluation and prediction

```yaml
general:
  seed: 20
  device: "cuda" 
  num_classes: 5
  class_names: ["EO", "IO", "LO", "PTE", "IPTE"]

paths:
  data_dir: "../data/structured/" 
  output_dir: "./outputs_debug"
  model_save_path: "./outputs_debug/best_model.pth"
  log_dir: "./logs_debug"
  eval_log_dir: "./logs_eval_debug"

training:
  model_type: "scratch"
  epochs: 100  
  batch_size: 8 
  learning_rate: 1e-3
  weight_decay: 0.0
  scheduler: true
  step_size: 10
  gamma: 0.1
  early_stopping: false  
  patience: 3

dataset:
  img_size: 224        
  val_split: 0.2
  test_split: 0.1
  augmentations: true     

evaluate:
  model_path: "../outputs/best_model.pth"   
  log_dir: "./logs_eval_debug"

test: 
  output_res_path: "../test_outputs/"
  model_path: C:/Users/guill/Documents/github_projects/neuroflux-classification/src/outputs_debug/best_model.pth
  input_dir: "../data/raw/IO/"
  output_csv:  "../test/"
  model_type: "scratch"
```

## Results 

 -- TODO: explain network architecture and metric selection + results

## License 
MIT License. Feel free to use and modify this code for academic or research purposes.

