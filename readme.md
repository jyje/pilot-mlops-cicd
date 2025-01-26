# pilot-mlops-cicd
This project implements a complete MLOps pipeline from model training to service deployment in a Cloud Native environment.

## Project Goals

### MLOps Pipeline Implementation
- Build a CI/CD system for model training and deployment
- Persistent Storage for datasets and models
- Implement a model training and export workflow in Kubernetes environments

## Tech Stack

### Infrastructure & Deployment
- Kubernetes & Helm
- CI: GitHub Actions
- CD: GitOps with ArgoCD

### ML Components
- Base Model: ResNet-18
- Dataset: MNIST
- Language: Python
- Serving Platform: NVIDIA Triton Inference Server
