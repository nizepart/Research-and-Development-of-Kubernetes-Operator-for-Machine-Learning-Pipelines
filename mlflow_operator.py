import kopf
import kubernetes
import asyncio
import logging

from kubernetes import config
from kubernetes.client.rest import ApiException

from mlflow.tracking import MlflowClient

from prometheus_api_client import PrometheusConnect

config.load_incluster_config()

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)

def extract_relative_path(source_uri):
    if source_uri.startswith('mlflow-artifacts:/'):
        relative_path = source_uri.replace('mlflow-artifacts:/', '', 1)
    else:
        relative_path = source_uri
    relative_path = relative_path.lstrip('/')
    return relative_path

@kopf.on.create('mlflow.nizepart.com', 'v1alpha1', 'mlflowmodels')
@kopf.on.update('mlflow.nizepart.com', 'v1alpha1', 'mlflowmodels')
async def mlflowmodel_handler(spec, name, namespace, status, body, **kwargs):
    model_name = spec.get('modelName')
    model_alias = spec.get('modelAlias')
    monitoring_interval = spec.get('monitoringInterval', 60)
    minio_secret_name = spec.get('minioSecret')

    # Initialize Kubernetes client
    custom_api = kubernetes.client.CustomObjectsApi()

    # Set up a logger for this model
    logger = logging.getLogger(f"{name}-{namespace}")

    # Configure the logger if needed (e.g., set level, add handlers)
    logger.setLevel(logging.INFO)

    # Initialize MLflow client
    mlflow_client = MlflowClient()

    # Prometheus URL
    prometheus_url = "http://seldon-monitoring-prometheus.seldon-monitoring.svc.cluster.local:9090"

    # Get current model versions from status
    current_model_version = None
    previous_model_version = None
    if status is not None:
        current_model_version = status.get('currentModelVersion')
        previous_model_version = status.get('previousModelVersion')

    while True:
        # Attempt to get model version by alias
        try:
            model_version = mlflow_client.get_model_version_by_alias(model_name, model_alias)
            alias_exists = True
        except Exception as e:
            alias_exists = False

        if not alias_exists:
            # Update status with error
            body_status = {
                "status": {
                    "error": f"Alias '{model_alias}' does not exist",
                    "currentModelVersion": None,
                    "previousModelVersion": None
                }
            }
            custom_api.patch_namespaced_custom_object_status(
                group='mlflow.nizepart.com',
                version='v1alpha1',
                namespace=namespace,
                plural='mlflowmodels',
                name=name,
                body=body_status
            )

            # Delete SeldonDeployment if it exists
            await delete_seldon_deployment(name, namespace)

            current_model_version = None
            previous_model_version = None

            # Log and add event
            logger.error(f"[{namespace}/{name}] Alias '{model_alias}' does not exist.")
            kopf.event(body, type="Warning", reason="AliasNotFound", message=f"Alias '{model_alias}' does not exist.")

            await asyncio.sleep(monitoring_interval)
            continue

        new_model_version = model_version.version

        if current_model_version != new_model_version:
            # New model version detected

            # Update versions in status
            previous_model_version = current_model_version
            current_model_version = new_model_version

            body_status = {
                "status": {
                    "currentModelVersion": current_model_version,
                    "previousModelVersion": previous_model_version,
                    "error": None
                }
            }
            custom_api.patch_namespaced_custom_object_status(
                group='mlflow.nizepart.com',
                version='v1alpha1',
                namespace=namespace,
                plural='mlflowmodels',
                name=name,
                body=body_status
            )

            # Log and add event
            logger.info(f"[{namespace}/{name}] New model version detected: {current_model_version}")
            kopf.event(body, type="Normal", reason="NewModelVersionDetected", message=f"New model version {current_model_version} detected.")

            # Update SeldonDeployment
            base_uri = "s3://mlflow"
            relative_path = extract_relative_path(model_version.source)
            new_model_uri = f"{base_uri}/{relative_path}"

            # Get URI of the previous model if it exists
            if previous_model_version is not None:
                previous_model_version_info = mlflow_client.get_model_version(model_name, previous_model_version)
                relative_path_prev = extract_relative_path(previous_model_version_info.source)
                old_model_uri = f"{base_uri}/{relative_path_prev}"
            else:
                old_model_uri = None

            # Create or update SeldonDeployment with two predictors
            await create_or_update_seldon_deployment(
                name=name,
                namespace=namespace,
                minio_secret_name=minio_secret_name,
                body=body,
                current_model_version=current_model_version,
                previous_model_version=previous_model_version,
                new_model_uri=new_model_uri,
                old_model_uri=old_model_uri,
                prometheus_url=prometheus_url,
                logger=logger  # Pass the logger
            )
        else:
            # No changes, continue monitoring
            pass

        await asyncio.sleep(monitoring_interval)

async def create_or_update_seldon_deployment(name, namespace, minio_secret_name, body, current_model_version,
                                             previous_model_version, new_model_uri, old_model_uri, prometheus_url, logger):
    # Get UID of the MlflowModel resource to set ownerReferences
    owner_uid = body['metadata']['uid']

    # Set ownerReferences
    owner_reference = [{
        "apiVersion": "mlflow.nizepart.com/v1alpha1",
        "kind": "MlflowModel",
        "name": name,
        "uid": owner_uid,
        "controller": True,
        "blockOwnerDeletion": True
    }]

    # Initialize Prometheus client
    prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)

    # Define threshold values for metrics
    thresholds = {
        'latency_95th': 0.05,  # Allowable increase of 5%
        'error_rate': 0.02,    # Allowable increase of 2%
        'latency_avg': 0.05    # Allowable increase of 5%
    }

    predictors = []

    # Initial traffic values
    if previous_model_version is not None:
        # Two predictors, start with minimal traffic to the new model
        traffic_prev = 90
        traffic_current = 10
    else:
        # Only one predictor, traffic should be 100%
        traffic_current = 100
        traffic_prev = 0

    # Predictor for the previous version
    if previous_model_version is not None:
        predictor_prev = {
            "graph": {
                "name": f"classifier-{previous_model_version}",
                "implementation": "MLFLOW_SERVER",
                "modelUri": old_model_uri,
                "envSecretRefName": minio_secret_name,
                "children": []
            },
            "name": f"v{previous_model_version}",
            "replicas": 1,
            "traffic": traffic_prev
        }
        predictors.append(predictor_prev)

    # Predictor for the current version
    predictor_current = {
        "graph": {
            "name": f"classifier-{current_model_version}",
            "implementation": "MLFLOW_SERVER",
            "modelUri": new_model_uri,
            "envSecretRefName": minio_secret_name,
            "children": []
        },
        "name": f"v{current_model_version}",
        "replicas": 1,
        "traffic": traffic_current
    }
    predictors.append(predictor_current)

    # Define SeldonDeployment
    seldon_deployment = {
        "apiVersion": "machinelearning.seldon.io/v1",
        "kind": "SeldonDeployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "ownerReferences": owner_reference
        },
        "spec": {
            "name": name,
            "protocol": "kfserving",
            "predictors": predictors
        }
    }

    # Apply SeldonDeployment
    apps_api = kubernetes.client.CustomObjectsApi()

    # Function to update SeldonDeployment
    async def apply_seldon_deployment(seldon_deployment):
        try:
            # Check if SeldonDeployment already exists
            existing_sd = apps_api.get_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1",
                namespace=namespace,
                plural="seldondeployments",
                name=name
            )

            # Get resourceVersion from the existing resource
            resource_version = existing_sd.get("metadata", {}).get("resourceVersion")
            if resource_version:
                # Set resourceVersion in the updated object
                seldon_deployment["metadata"]["resourceVersion"] = resource_version

            # Update SeldonDeployment
            apps_api.replace_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1",
                namespace=namespace,
                plural="seldondeployments",
                name=name,
                body=seldon_deployment
            )
        except ApiException as e:
            if e.status == 404:
                # Create SeldonDeployment
                apps_api.create_namespaced_custom_object(
                    group="machinelearning.seldon.io",
                    version="v1",
                    namespace=namespace,
                    plural="seldondeployments",
                    body=seldon_deployment
                )
            else:
                logger.error(f"[{namespace}/{name}] Error applying SeldonDeployment: {e}")
                raise e

    # Apply the initial SeldonDeployment
    await apply_seldon_deployment(seldon_deployment)
    logger.info(f"[{namespace}/{name}] Applied initial SeldonDeployment.")

    # If there is a previous version, start gradient traffic shifting
    if previous_model_version is not None:
        max_traffic = 100
        step = 10    # Traffic change step (in percent)
        interval = 60  # Interval in seconds between traffic increases
        max_attempts = 10  # Number of attempts per iteration
        attempt_delay = 10  # Delay in seconds between attempts within an iteration

        while traffic_current < max_traffic:
            attempt = 0
            promotion_success = False

            while attempt < max_attempts:
                # Collect metrics for the new and old models
                new_metrics = get_model_metrics(prom, name, f"v{current_model_version}", namespace)
                old_metrics = get_model_metrics(prom, name, f"v{previous_model_version}", namespace)

                # Log metrics
                logger.info(f"[{namespace}/{name}] Metrics for new model (version {current_model_version}): {new_metrics}")
                logger.info(f"[{namespace}/{name}] Metrics for old model (version {previous_model_version}): {old_metrics}")

                # Decide whether to increase traffic
                if should_promote_model(new_metrics, old_metrics, thresholds, logger):
                    # Increase traffic to new model
                    traffic_current += step
                    traffic_prev -= step

                    # Cap traffic at maximum and minimum
                    traffic_current = min(traffic_current, max_traffic)
                    traffic_prev = max(traffic_prev, 0)

                    # Update traffic values in predictors
                    for predictor in predictors:
                        if predictor["name"] == f"v{current_model_version}":
                            predictor["traffic"] = traffic_current
                        elif predictor["name"] == f"v{previous_model_version}":
                            predictor["traffic"] = traffic_prev

                    # Update SeldonDeployment with new traffic values
                    seldon_deployment["spec"]["predictors"] = predictors
                    await apply_seldon_deployment(seldon_deployment)

                    logger.info(f"[{namespace}/{name}] Increased traffic to new model to {traffic_current}%")
                    # Add event
                    kopf.event(body, type="Normal", reason="TrafficIncrease", message=f"Increased traffic to new model to {traffic_current}%")
                    promotion_success = True
                    break  # Exit the attempt loop
                else:
                    # Metrics do not meet conditions, wait and retry
                    attempt += 1
                    if attempt < max_attempts:
                        logger.info(f"[{namespace}/{name}] Attempt {attempt}/{max_attempts}: Metrics do not meet conditions, retrying after {attempt_delay} seconds.")
                        await asyncio.sleep(attempt_delay)
                    else:
                        logger.warning(f"[{namespace}/{name}] Metrics did not meet conditions after {max_attempts} attempts, stopping promotion.")
                        # Add event
                        kopf.event(body, type="Warning", reason="PromotionFailed", message=f"Metrics did not meet conditions after {max_attempts} attempts, stopping promotion.")
                        # Implement rollback logic here if needed

            if not promotion_success:
                # Metrics did not meet conditions after max_attempts, stop promotion
                break  # Exit the traffic increase loop

            # Wait before the next traffic increase
            await asyncio.sleep(interval)

        # When the new model reaches 100% traffic, we can remove the previous model
        if traffic_current == max_traffic:
            predictors = [predictor_current]
            seldon_deployment["spec"]["predictors"] = predictors
            await apply_seldon_deployment(seldon_deployment)
            logger.info(f"[{namespace}/{name}] The new model has received 100% of traffic. Previous model has been removed.")
            # Add event
            kopf.event(body, type="Normal", reason="PromotionComplete", message="New model now receives 100% traffic. Previous model has been removed.")

def get_model_metrics(prom, deployment_name, predictor_name, namespace, elapsed_time=60):
    metrics = {}

    # 1. 95th percentile latency
    query_latency_95th = f"""histogram_quantile(0.95, sum(rate(seldon_api_executor_client_requests_seconds_bucket{{deployment_name="{deployment_name}", predictor_name="{predictor_name}", namespace="{namespace}"}}[{elapsed_time}s])) by (le))"""
    result = prom.custom_query(query_latency_95th)
    if result:
        metrics['latency_95th'] = float(result[0]['value'][1])
    else:
        metrics['latency_95th'] = None

    # 2. Number of error responses
    query_error_responses = f"""sum(increase(seldon_api_executor_server_requests_seconds_count{{code!="200", deployment_name="{deployment_name}", predictor_name="{predictor_name}", namespace="{namespace}"}}[{elapsed_time}s])) or on() vector(0)"""
    result = prom.custom_query(query_error_responses)
    if result:
        metrics['error_responses'] = float(result[0]['value'][1])
    else:
        metrics['error_responses'] = 0.0

    # 3. Fraction of requests with error responses
    query_total_responses = f"""sum(increase(seldon_api_executor_server_requests_seconds_count{{deployment_name="{deployment_name}", predictor_name="{predictor_name}", namespace="{namespace}"}}[{elapsed_time}s])) or on() vector(0)"""
    result_total = prom.custom_query(query_total_responses)
    total_responses = float(result_total[0]['value'][1]) if result_total else 0.0

    if total_responses > 0:
        metrics['error_rate'] = metrics['error_responses'] / total_responses
    else:
        metrics['error_rate'] = None

    # 4. Mean latency
    query_latency_avg_sum = f"""sum(increase(seldon_api_executor_client_requests_seconds_sum{{deployment_name="{deployment_name}", predictor_name="{predictor_name}", namespace="{namespace}"}}[{elapsed_time}s])) or on() vector(0)"""
    result_latency_sum = prom.custom_query(query_latency_avg_sum)
    latency_sum = float(result_latency_sum[0]['value'][1]) if result_latency_sum else 0.0

    query_latency_avg_count = f"""sum(increase(seldon_api_executor_client_requests_seconds_count{{deployment_name="{deployment_name}", predictor_name="{predictor_name}", namespace="{namespace}"}}[{elapsed_time}s])) or on() vector(0)"""
    result_latency_count = prom.custom_query(query_latency_avg_count)
    latency_count = float(result_latency_count[0]['value'][1]) if result_latency_count else 0.0

    if latency_count > 0:
        metrics['latency_avg'] = latency_sum / latency_count
    else:
        metrics['latency_avg'] = None

    # 5. Number of requests
    metrics['request_count'] = latency_count

    # 6. Number of feedback requests
    query_feedback_requests = f"""sum(increase(seldon_api_executor_server_requests_seconds_count{{service="feedback", deployment_name="{deployment_name}", predictor_name="{predictor_name}", namespace="{namespace}"}}[{elapsed_time}s])) or on() vector(0)"""
    result_feedback = prom.custom_query(query_feedback_requests)
    if result_feedback:
        metrics['feedback_request_count'] = float(result_feedback[0]['value'][1])
    else:
        metrics['feedback_request_count'] = 0.0

    return metrics

def should_promote_model(new_metrics, old_metrics, thresholds, logger):
    """
    Decides whether to promote the new model based on metrics.

    :param new_metrics: Metrics of the new model.
    :param old_metrics: Metrics of the old model.
    :param thresholds: Dictionary with threshold values for metrics.
    :param logger: Logger instance.
    :return: True if the model should be promoted, else False.
    """
    # Check that metrics are available
    required_metrics = ['latency_95th', 'error_rate', 'latency_avg']
    for metric in required_metrics:
        if new_metrics.get(metric) is None or old_metrics.get(metric) is None:
            logger.warning(f"Metric {metric} is not available for one of the models.")
            return False

    # Check promotion conditions
    promote = True

    # 1. Check 95th percentile latency
    if new_metrics['latency_95th'] <= old_metrics['latency_95th'] * (1 + thresholds['latency_95th']):
        logger.info("95th percentile latency of the new model is acceptable.")
    else:
        logger.warning("95th percentile latency of the new model exceeds the allowed threshold.")
        promote = False

    # 2. Check error rate
    if new_metrics['error_rate'] <= old_metrics['error_rate'] * (1 + thresholds['error_rate']):
        logger.info("Error rate of the new model is acceptable.")
    else:
        logger.warning("Error rate of the new model exceeds the allowed threshold.")
        promote = False

    # 3. Check average latency
    if new_metrics['latency_avg'] <= old_metrics['latency_avg'] * (1 + thresholds['latency_avg']):
        logger.info("Average latency of the new model is acceptable.")
    else:
        logger.warning("Average latency of the new model exceeds the allowed threshold.")
        promote = False

    return promote

async def delete_seldon_deployment(name, namespace):
    apps_api = kubernetes.client.CustomObjectsApi()
    try:
        apps_api.delete_namespaced_custom_object(
            group="machinelearning.seldon.io",
            version="v1",
            namespace=namespace,
            plural="seldondeployments",
            name=name,
            body=kubernetes.client.V1DeleteOptions()
        )
        logging.info(f"[{namespace}/{name}] SeldonDeployment '{name}' deleted from namespace '{namespace}'.")
    except ApiException as e:
        if e.status != 404:
            logging.error(f"[{namespace}/{name}] Error deleting SeldonDeployment '{name}': {e}")
            raise e
