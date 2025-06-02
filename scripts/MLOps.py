import os
from datetime import datetime # Import datetime for timestamp

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.sweep import Choice
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data
from azure.ai.ml.entities import AmlCompute # Import AmlCompute for creating compute

# Initialize MLClient
credential = DefaultAzureCredential()

# --- Explicitly get environment variables and check for existence ---
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group_name = os.getenv("AZURE_RESOURCE_GROUP")
# Ensure this matches the env var name set in your workflow
workspace_name = os.getenv("AZUREML_WORKSPACE_NAME")

if not subscription_id:
    raise ValueError("AZURE_SUBSCRIPTION_ID environment variable is not set.")
if not resource_group_name:
    raise ValueError("AZURE_RESOURCE_GROUP environment variable is not set.")
if not workspace_name:
    raise ValueError("AZUREML_WORKSPACE_NAME environment variable is not set or is empty. Please check your GitHub secrets and workflow 'env' block.")
# --- END Checks ---

ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group_name,
    workspace_name=workspace_name
)

# --- Ensure compute cluster exists ---
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
# --- END FIX ---


# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# --- IMPORTANT: Ensure the 'environment' field in these YAMLs is updated ---
# For example, in data_prep.yml, change:
# environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1
# To:
# environment: azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1
# Apply similar changes to train_step.yml and model_register_component.yml
# ---
step_process = load_component(source=os.path.join(base_dir, "../components/data_prep.yml"))
train_step = load_component(source=os.path.join(base_dir, "../components/train_step.yml"))
model_register_component = load_component(source=os.path.join(base_dir, "../components/model_register_component.yml"))

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

    # --- FIX: Temporarily disable model_register_step inside the pipeline due to path resolution issue ---
    # We will register the model in a separate step after the pipeline completes.
    # model_register_step = model_register_component(model=sweep_job.outputs.model_output)

    return {
        "pipeline_job_train_data": preprocess_step.outputs.train_data,
        "pipeline_job_test_data": preprocess_step.outputs.test_data,
        "pipeline_job_best_model": sweep_job.outputs.model_output,
        "pipeline_job_best_run_id": sweep_job.outputs.best_child_run_id, # EXPOSE best_child_run_id as pipeline output
    }

# --- Generate a dynamic version based on current timestamp ---
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

# --- FIX: After pipeline completes, get the best run ID and register model ---
# This part executes only after the entire pipeline finishes.
print(f"Pipeline job '{pipeline_job.name}' completed.")
best_run_id = pipeline_job.outputs.get("pipeline_job_best_run_id")

if best_run_id:
    print(f"Best run ID from sweep: {best_run_id}")
    # Now, use model_register.py (or a similar script) to register the model
    # from this run ID. This avoids the uri_folder binding issue.
    # We will call model_register.py as a standalone script here.
    # This requires model_register.py to be adjusted to take run_id as input.

    print("Attempting to register model using the best run ID.")
    try:
        # Re-initialize argparse for model_register.py if it's called as a function
        # Or, just use mlflow.register_model directly with the best_run_id
        # Ensure mlflow is configured to use the Azure ML tracking URI
        mlflow.set_tracking_uri(ml_client.tracking_uri)
        
        model_uri = f"runs:/{best_run_id}/artifacts/model" # Assuming model artifact path is 'model'
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
    print("Could not retrieve best run ID from pipeline outputs.")
# --- END FIX ---


# Output results (these will be for the overall pipeline job)
print(f"Train data location: {pipeline_job.outputs['pipeline_job_train_data']}")
print(f"Test data location: {pipeline_job.outputs['pipeline_job_test_data']}")
print(f"Best model location: {pipeline_job.outputs['pipeline_job_best_model']}")
# print(f"Best run ID: {pipeline_job.outputs['pipeline_job_best_run_id']}") # This is already printed above
