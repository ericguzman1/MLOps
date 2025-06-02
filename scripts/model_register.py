# scripts/model_register.py
import argparse
import mlflow
import os
from pathlib import Path # Import Path for robust path manipulation

def main():
    parser = argparse.ArgumentParser(description="Model registration script.")
    parser.add_argument("--model_path", type=str, help="Path to the trained model artifact (URI folder).")
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("Model path is required for model registration.")

    # --- FIX: Robustly clean and validate the received model path ---
    # Convert to Path object for easier manipulation and normalization
    received_model_path = Path(args.model_path)

    # Normalize the path (resolves '..' and cleans up trailing slashes, etc.)
    cleaned_model_path = received_model_path.resolve()

    # Check for the problematic placeholder
    if "${{name}}" in str(cleaned_model_path):
        raise ValueError(
            f"Model path still contains unexpanded Azure ML placeholder: '{cleaned_model_path}'. "
            "This indicates an issue with Azure ML's internal path resolution for MLflow models from sweep outputs. "
            "Please report this to Azure ML support or consider alternative model passing strategies."
        )

    # MLflow expects the path to the directory containing the MLmodel file
    # Sometimes, the mounted path might point to a parent directory, or include a trailing '.'
    # We should ensure it points to the actual MLflow model directory if structure is like 'path/model/'
    # You might need to adjust this if your model is always in a specific subfolder like 'model/'
    # For now, we assume cleaned_model_path is the root of the MLflow model artifact.

    print(f"Attempting to register model from local path: {cleaned_model_path}")
    
    # Ensure the path exists locally before trying to register
    if not cleaned_model_path.exists():
        raise FileNotFoundError(f"Cleaned model path does not exist: '{cleaned_model_path}'")
    if not cleaned_model_path.is_dir():
        raise ValueError(f"Cleaned model path is not a directory: '{cleaned_model_path}'")
    if not (cleaned_model_path / "MLmodel").exists():
        raise ValueError(f"MLmodel file not found in directory: '{cleaned_model_path}'")


    # Use mlflow.register_model directly with the local file:// URI
    # This will register the model that is locally available at cleaned_model_path
    try:
        registered_model_name = "trained_decision_tree_model"
        
        registered_model = mlflow.register_model(
            model_uri=f"file://{cleaned_model_path}", # Use file:// URI for local path
            name=registered_model_name,
            # MLflow will auto-increment version if not specified
            tags={"registered_by_aml_pipeline": True}
        )
        print(f"Model registered successfully: Name='{registered_model.name}', Version='{registered_model.version}'")
    except Exception as e:
        # Re-raise the exception to make sure the job fails
        print(f"Error registering model from path {cleaned_model_path}: {e}")
        raise

if __name__ == "__main__":
    main()
