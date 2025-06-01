import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input, load_component
from azure.ai.ml.sweep import Choice
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import Data

# Initialize MLClient
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZUREML_WORKSPACE_NAME")
)

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
        criterion="${{search_space.criterion}}",
        max_depth="${{search_space.max_depth}}"
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

    model_register_step = model_register_component(model=sweep_job.outputs.model_output)

    return {
        "pipeline_job_train_data": preprocess_step.outputs.train_data,
        "pipeline_job_test_data": preprocess_step.outputs.test_data,
        "pipeline_job_best_model": sweep_job.outputs.model_output,
    }
# Create and register the dataset
data_asset = Data(
    name="used-cars-data",
    version="23",
    type="uri_file",
    path="data/used_cars.csv"
)
ml_client.data.create_or_update(data_asset)
# Get data path from Azure ML dataset
data_path = ml_client.data.get("used-cars-data", version="23").path

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
