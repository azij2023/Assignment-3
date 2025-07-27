Assignment 3: End-to-End MLOps Pipeline

This project implements a complete MLOps workflow from model training to deployment and optimization using manual quantization techniques.

 Repository Structure

```bash
Assignment-3/
├── train.py              # Trains a scikit-learn LinearRegression model
├── quantize.py           # Manually quantizes trained parameters
├── inference.py          # Performs inference using de-quantized weights
├── Dockerfile            # Container setup for testing model pipeline
├── predict.py            # Validates Docker container by running predictions
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI/CD workflow
└── README.md             # You're reading it!
 Branching Strategy
main: Initial setup with README and .gitignore

dev: Training LinearRegression model and saving via joblib

docker_ci: Containerized pipeline and CI/CD workflow

quantization: Manual quantization and inference with PyTorch

Model Summary
Metric	Sklearn Model	Quantized Model
R² Score	Your value	Your value
Model Size (KB)	Your file size	Your file size
 Workflow Overview
 Step 1: Model Training
Used California Housing dataset

Trained LinearRegression model using scikit-learn

Saved model to model.joblib

Step 2: Containerization & CI/CD
Dockerfile builds an image for prediction

CI workflow (ci.yml) runs:

Train script

Build and run container

Validate with predict.py

Push image to Docker Hub

 Step 3: Manual Quantization
Extracted .coef_ and .intercept_ from scikit model

Converted weights to unsigned 8-bit integers

Saved both unquantized and quantized parameters

Performed inference using de-quantized weights

 Submission Links
GitHub Repository

Docker Hub Repository

 Screenshot Guide
Include screenshots for:

Branch creation

File commits and push logs

CI/CD run status

Quantization output and inference results

 Evaluation Points
 Functional CI/CD pipeline with Docker

 Modular branching and isolated development

 Manual quantization process and inference flow


Comparison Table
Metric	Original Sklearn Model	Quantized Model
R² Score	0..5928	-4.1142
Model Size	0. 40 KB	0..48 KB
