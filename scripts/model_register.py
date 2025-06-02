# scripts/model_register.py
import argparse
import mlflow
import os

def main():
    parser = argparse.ArgumentParser(description="Model registration script.")
    # FIX: Change argument to accept a model path (uri_folder) instead of run_id
    parser.add_argument("--model_path", type=str, help="Path to the trained model artifact (URI folder).")
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Model path is required for model registration.")

    # FIX: Load the model directly from the provided path
    # The model path is a local directory where the model artifact is mounted by Azure ML
    print(f"Attempting to load model from local path: {args.model_path}")
    
    # Ensure MLflow is configured to use the Azure ML tracking URI
    # This is often handled automatically in an Azure ML job, but explicit setting can help
    # mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI")) # This might be needed if not set automatically

    try:
        # Load the MLflow model from the local path
        model = mlflow.sklearn.load_model(args.model_path)
        print("Model loaded successfully from local path.")
    except Exception as e:
        print(f"Error loading model from path {args.model_path}: {e}")
        raise # Re-raise the exception if model loading fails

    # Register the model
    registered_model_name = "trained_decision_tree_model"
    
    try:
        # Registering the model directly from the loaded model object
        # MLflow will automatically log it to the current run and then register it
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{mlflow.active_run().info.run_id}/artifacts/model", # Use current run's artifact URI
            name=registered_model_name,
            # No version specified, MLflow will auto-increment
            tags={"registered_by_aml_pipeline": True}
        )
        print(f"Model registered successfully: Name='{registered_model.name}', Version='{registered_model.version}'")
    except Exception as e:
        print(f"Error registering model: {e}")
        raise # Re-raise the exception if registration fails

if __name__ == "__main__":
    main()
