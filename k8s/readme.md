# k8s

## Requirements

- OS: debian/ubuntu, darwin/macOS
- Container runtime: Any container runtime (e.g. docker, containerd etc.), platform (e.g. docker-desktop, orbstack, etc.)
- CLI: kubectl
- Kubernetes Platform
    - minikube (verified)
    - microk8s (verified)
    - orbstack (verified)
    - or any other Kubernetes platform with StorageClass


## Getting Started
On Kubernetes platform with StorageClass

```sh
kubectl config use-context minikube 
# or orbstack, docker-desktop, microk8s

git clone -b develop https://github.com/jyje/pilot-mlops-cicd.git
cd pilot-mlops-cicd/k8s/pilot
kubectl create namespace pilot
kubectl apply -n pilot -f .
```

## Step-by-step
### 1. Run the model builder

```sh
git clone -b develop https://github.com/jyje/pilot-mlops-cicd.git
cd pilot-mlops-cicd/k8s/pilot
kubectl create namespace pilot
kubectl apply -n pilot -f pilot-persistence.yaml
kubectl apply -n pilot -f pilot-model-builder.yaml
```

### 2. Run the model server

```sh
kubectl apply -n pilot -f pilot-model-server.yaml
```

### 3. Run the model client

```sh
kubectl apply -n pilot -f pilot-model-client.yaml
```
