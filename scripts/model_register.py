# scripts/model_register.py
import argparse
import mlflow
import os

def main():
    parser = argparse.ArgumentParser(description="Model registration script.")
    parser.add_argument("--run_id", type=str, help="MLflow Run ID of the best trained model.")
    args = parser.parse_args()

    if not args.run_id:
        raise ValueError("MLflow Run ID is required for model registration.")

    # Construct the MLflow Model URI
    # 'artifacts/model' is the conventional path where mlflow.sklearn.log_model saves the model
    model_uri = f"runs:/{args.run_id}/artifacts/model"
    print(f"Attempting to register model from MLflow URI: {model_uri}")

    # Initialize MLflow with Azure ML tracking URI (optional, usually handled by environment context in AML)
    # mlflow.set_tracking_uri(mlflow.get_tracking_uri()) # This line is often not needed if running inside an AML job

    # Register the model
    # The name of the registered model in Azure ML Model Registry
    registered_model_name = "trained_decision_tree_model"
    
    # Check if the model already exists in the registry to avoid re-registration if desired
    # For CI/CD, often a new version is created.
    try:
        # Registering with a new version each time for CI/CD robustness
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_model_name,
            # No version specified, MLflow will auto-increment
            tags={"source_pipeline_run_id": args.run_id, "registered_by_aml_pipeline": True}
        )
        print(f"Model registered successfully: Name='{registered_model.name}', Version='{registered_model.version}'")
    except Exception as e:
        print(f"Error registering model: {e}")
        # If you want to force registration, you can potentially try an update here,
        # but typically for CI/CD, you register new versions.
        raise # Re-raise the exception if registration fails

if __name__ == "__main__":
    main()
