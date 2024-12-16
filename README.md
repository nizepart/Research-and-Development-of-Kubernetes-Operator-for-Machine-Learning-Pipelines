# MLFlow Operator

This project provides an **MLFlow Operator** to integrate MLFlow with Kubernetes-based ML orchestration tools like **Seldon Core**. The operator monitors MLFlow for new or updated model versions and automatically creates or updates Seldon Core deployments.

## Features

- Automates model deployment by listening for updates in MLFlow models.
- Configures and updates Kubernetes Custom Resource Definitions (CRDs).
- Supports model traffic promotion and rollback based on Prometheus monitoring.
- Gradual traffic shifting between model versions with metrics validation.

---

## File Structure

### **`mlflow_operator.py`**
The core logic of the operator, implemented using [Kopf](https://kopf.readthedocs.io/).  
Features include:
- Watching and handling `MlflowModel` resources.
- Creating/Updating Seldon Core deployments.
- Monitoring Prometheus metrics to ensure model quality before traffic promotion.

### **`crd.yaml`**
Defines the `MlflowModel` Custom Resource Definition (CRD).  
Key specifications:
- `spec` fields: `modelName`, `modelAlias`, `monitoringInterval`, and `minioSecret`.
- `status` fields: Tracks `currentModelVersion`, `previousModelVersion`, and error info.

### **`mlflow-operator-deployment.yaml`**
Kubernetes deployment configuration for the MLFlow Operator.  
Key points:
- Runs the operator as a container using the image `nizepart/mlflow-operator:latest`.
- Deploys in the `mlflow-operator` namespace.

### **`rbac.yaml`**
Role-based access control (RBAC) setup for the operator.  
Includes:
- `ServiceAccount` for the operator.
- Cluster-level permissions to manage `MlflowModel` resources, secrets, events, namespaces, and Seldon Core deployments.

---

## Deployment Instructions

1. **Apply the CRD**
   ```bash
   kubectl apply -f crd.yaml
   ```

2. **Set Up RBAC**
   ```bash
   kubectl apply -f rbac.yaml
   ```

3. **Deploy the Operator**
   ```bash
   kubectl apply -f mlflow-operator-deployment.yaml
   ```

4. **Monitor Logs**
   To verify the operator is running successfully:
   ```bash
   kubectl logs -l app=mlflow-operator -n mlflow-operator
   ```

---

## How It Works

1. **CRD Integration:**
   The operator listens to `MlflowModel` CRDs created in the cluster.

2. **Model Detection:**
   It uses MLFlow APIs to monitor model versions tagged with specific aliases.

3. **Seldon Deployment:**
   Automatically generates or updates Seldon Core deployments with the new model.

4. **Traffic Gradual Shifting:**
   Shifts traffic incrementally from the previous to the new model based on Prometheus metrics, ensuring reliability.

---

## Requirements

- **Kubernetes cluster** (v1.20+)
- **MLFlow** instance for model versioning.
- **Seldon Core** for model inference.
- **Istio** configured as the service mesh for Seldon Core.
- **Prometheus** for monitoring metrics.

---

## Contributions

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.
