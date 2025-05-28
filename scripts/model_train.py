import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.start_run()  # Start the MLflow experiment run

os.makedirs("./outputs", exist_ok=True)  # Create the "outputs" directory if it doesn't exist

def select_first_file(path):
    """Selects the first file in a folder, assuming there's only one file.
    Args:
        path (str): Path to the directory or file to choose.
    Returns:
        str: Full path of the selected file.
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument('--criterion', type=str, default='squared_error',
                        help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')
    parser.add_argument("--model_output", type=str, help="Path of output model")
    args = parser.parse_args()

    # Load datasets
    train_df = pd.read_csv(select_first_file(args.train_data))
    test_df = pd.read_csv(select_first_file(args.test_data))

    # Strip whitespace from column names
    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    # Debug: Print column names
    print("Columns in train_df:", train_df.columns.tolist())

    # Dropping the label column and assigning it to y_train
    y_train = train_df["price"].values  # 'Price' is the target variable in this case study

    # Dropping the 'Price' column from train_df to get the features and converting to array for model training
    X_train = train_df.drop("price", axis=1).values

    # Dropping the label column and assigning it to y_test
    y_test = test_df["price"].values  # 'Price' is the target variable for testing

    # Dropping the 'Price' column from test_df to get the features and converting to array for model testing
    X_test = test_df.drop("price", axis=1).values

    # Initialize and train a decision tree regressor
    tree_model = DecisionTreeRegressor(criterion=args.criterion, max_depth=args.max_depth)
    tree_model = tree_model.fit(X_train, y_train)
    tree_predictions = tree_model.predict(X_test)

    # Compute and log regression metrics
    mse = mean_squared_error(y_test, tree_predictions)
    r2 = r2_score(y_test, tree_predictions)

    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

    mlflow.log_metric("MSE", float(mse))
    mlflow.log_metric("R2", float(r2))

    # Output the trained model
    mlflow.sklearn.save_model(tree_model, args.model_output)

    mlflow.end_run()  # Ending the MLflow experiment run

if __name__ == "__main__":
    main()

