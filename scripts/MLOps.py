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
workspace_name = os.getenv("AZUREML_WORKSPACE_NAME") # Ensure this matches the env var name set in your workflow

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
        # criterion and max_depth will be provided by the sweep's search_space
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

    # --- FIX: Pass best_child_run_id instead of model output URI ---
    model_register_step = model_register_component(run_id=sweep_job.outputs.best_child_run_id)

    return {
        "pipeline_job_train_data": preprocess_step.outputs.train_data,
        "pipeline_job_test_data": preprocess_step.outputs.test_data,
        "pipeline_job_best_model": sweep_job.outputs.model_output, # Still return the model output URI if needed
        "pipeline_job_best_run_id": sweep_job.outputs.best_child_run_id, # FIX: Expose best_run_id as pipeline output
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

# Output results
print(f"Train data location: {pipeline_job.outputs['pipeline_job_train_data']}")
print(f"Test data location: {pipeline_job.outputs['pipeline_job_test_data']}")
print(f"Best model location: {pipeline_job.outputs['pipeline_job_best_model']}")
print(f"Best run ID: {pipeline_job.outputs['pipeline_job_best_run_id']}") # FIX: Print best run ID
