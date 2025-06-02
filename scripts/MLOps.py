import os
from datetime import datetime 
import mlflow 
from mlflow.entities import ViewType 

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.sweep import Choice
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import AmlCompute 

# Initialize MLClient
credential = DefaultAzureCredential()

# --- Explicitly get environment variables and check for existence ---
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")

if not subscription_id:
    raise ValueError("AZURE_SUBSCRIPTION_ID environment variable is not set.")
if not resource_group_name:
    raise ValueError("AZURE_RESOURCE_GROUP environment variable is not set.")
if not workspace_name:
    raise ValueError("AZUREML_WORKSPACE_NAME environment variable is not set or is empty. Please check your GitHub secrets and workflow 'env' block.")


ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name
)

compute_name = "cpu-cluster"
try:
    print(f"Checking if compute target '{compute_name}' exists...")
    ml_client.compute.get(name=compute_name)
    print(f"Compute target '{compute_name}' already exists.")
except Exception as e:
    print(f"Compute target '{compute_name}' not found. Creating a new one...")
    compute_config = AmlCompute(
        name=compute_name,
        type="amlcompute",
        size="STANDARD_DS3_V2",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120
    )
    ml_client.compute.begin_create_or_update(compute_config).wait()
    print(f"Compute target '{compute_name}' created successfully.")


# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

step_process = load_component(source=os.path.join(base_dir, "../components/data_prep.yml"))
train_step = load_component(source=os.path.join(base_dir, "../components/train_step.yml"))
model_register_component = load_component(source=os.path.join(base_dir, "../components/model_register.yml"))


# Define pipeline
@pipeline(compute="cpu-cluster", description="Pipeline for data preparation, training, and model registration")
def complete_pipeline(input_data_uri, test_train_ratio):
    preprocess_step = step_process(
        data=input_data_uri,
        test_train_ratio=test_train_ratio
    )

    training_job = train_step(
        train_data=preprocess_step.outputs.train_data,
        test_data=preprocess_step.outputs.test_data,
    )

    sweep_job = training_job.sweep(
        sampling_algorithm="random",
        primary_metric="r2_score",
        goal="maximize",
        search_space={
            "criterion": Choice(["squared_error", "absolute_error"]),
            "max_depth": Choice([3, 5, 10])
        }
    )

    sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)

    # model_register_step = model_register_component(model=sweep_job.outputs.model_output)

    return {
        "pipeline_job_train_data": preprocess_step.outputs.train_data,
        "pipeline_job_test_data": preprocess_step.outputs.test_data,
        "pipeline_job_best_model": sweep_job.outputs.model_output,
    }
    
current_time_version = datetime.now().strftime("%Y%m%d%H%M%S")

# Create and register the dataset
data_asset = Data(
    name="used-cars-data",
    version=current_time_version,
    type="uri_file",
    path="data/used_cars.csv"
)
ml_client.data.create_or_update(data_asset)

# Get data path from Azure ML dataset
data_path = ml_client.data.get("used-cars-data", version=current_time_version).path

# Create pipeline instance
pipeline_instance = complete_pipeline(
    input_data_uri=Input(type="uri_file", path=data_path),
    test_train_ratio=0.25
)

# Submit pipeline job
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_instance,
    experiment_name="decision_tree_training_pipeline"
)

# Stream job logs
ml_client.jobs.stream(pipeline_job.name)

# --- FIX: After pipeline completes, query for the best run ID and register model ---
print(f"Pipeline job '{pipeline_job.name}' completed.")

# Initialize MLflow client with the same tracking URI as MLClient
mlflow.set_tracking_uri(ml_client.tracking_uri)
mlflow_client = mlflow.tracking.MlflowClient()

best_run_id = None
try:
    # Get the overall pipeline run (which is a parent run for the sweep)
    # The pipeline_job.name is the run_id of the parent pipeline run
    parent_run = mlflow_client.get_run(pipeline_job.name)

    # Search for child runs of this pipeline run (these include the sweep trials)
    # Filter by experiment ID and parent_run_id
    child_runs = mlflow_client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{pipeline_job.name}'",
        order_by=["metrics.r2_score DESC"], # Order by primary metric (maximize)
        run_view_type=ViewType.ACTIVE_ONLY,
    )

    if child_runs:
        # The first run in the ordered list should be the best one
        best_child_run = child_runs[0]
        best_run_id = best_child_run.info.run_id
        print(f"Found best child run ID: {best_run_id} with r2_score: {best_child_run.data.metrics.get('r2_score')}")
    else:
        print("No child runs found for the pipeline job.")

except Exception as e:
    print(f"Error querying MLflow runs to find best child: {e}")
    raise

if best_run_id:
    print("Attempting to register model using the best run ID.")
    try:
        model_uri = f"runs:/{best_run_id}/artifacts/model" 
        registered_model_name = "trained_decision_tree_model"

        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
            tags={"source_pipeline_run_id": best_run_id, "registered_by_aml_pipeline": True}
        )
        print(f"Model registered successfully: Name='{registered_model.name}', Version='{registered_model.version}'")
    except Exception as e:
        print(f"Error during post-pipeline model registration: {e}")
        raise
else:
    print("Could not retrieve best run ID from pipeline outputs for model registration.")

# Output results (these will be for the overall pipeline job)
print(f"Train data location: {pipeline_job.outputs['pipeline_job_train_data']}")
print(f"Test data location: {pipeline_job.outputs['pipeline_job_test_data']}")
print(f"Best model location: {pipeline_job.outputs['pipeline_job_best_model']}")
