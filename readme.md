# pilot-mlops-cicd
This project implements a complete MLOps pipeline from model training to service deployment in a Cloud Native environment.

## Project Goals

### MLOps Pipeline Implementation
- Build a CI/CD system for model training and deployment
- Persistent Storage for datasets and models
- Implement a model training and export workflow in Kubernetes environments

## Tech Stack

### Infrastructure & Deployment
- Kubernetes
- Workload Manifests

#### Plan (NOT YET 🥲)
- Helm Charts
- CI: GitHub Actions
- CD: GitOps with ArgoCD

### ML Components
- Base Model: ResNet-18
- Dataset: MNIST
- Language: Python
- Serving Platform: NVIDIA Triton Inference Server

## References
### Official
- https://github.com/pytorch/serve
- https://github.com/pytorch/torchscript
- https://github.com/triton-inference-server/server
- https://github.com/triton-inference-server/client
- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags

### Community
- https://github.com/YH-Wu/Triton-Inference-Server-on-Kubernetes/blob/22.04/README.md
- https://velog.io/@kcw4875/Triton-Inference-Server-%EB%B6%80%EC%88%98%EA%B8%B0
- https://one-way-people.tistory.com/42