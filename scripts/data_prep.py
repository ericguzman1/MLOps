# scripts/data_prep.py
import pandas as pd
import numpy as np
import os
import argparse # Import argparse

def main():
    # 1. Create an ArgumentParser
    parser = argparse.ArgumentParser(description="Data preparation script.")
    parser.add_argument("--data", type=str, help="Path to the input CSV data.")
    parser.add_argument("--test_train_ratio", type=float, default=0.25, help="Ratio for test/train split.")
    parser.add_argument("--train_data", type=str, help="Path to save processed training data.")
    parser.add_argument("--test_data", type=str, help="Path to save processed testing data.")

    # 2. Parse arguments
    args = parser.parse_args()

    # Get input data path
    input_data_path = args.data

    # Log the input path for debugging purposes
    print(f"Reading data from: {input_data_path}")

    # 3. Read the data from the provided input path
    # Ensure to use the absolute path from the input argument
    df = pd.read_csv(input_data_path, delimiter='\t')

    # Basic data cleaning and preparation (adjust as per your actual data_prep.py logic)
    # Example: fill missing values for numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean()) # Fill with mean or other strategy

    # Example: handle categorical columns (one-hot encoding)
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=False)


    # Split data
    from sklearn.model_selection import train_test_split
    X = df.drop('price', axis=1) # Assuming 'price' is your target
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_train_ratio, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Ensure output directories exist
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    # Save processed data to the output paths provided by Azure ML
    train_output_path = os.path.join(args.train_data, 'train.csv')
    test_output_path = os.path.join(args.test_data, 'test.csv')

    print(f"Saving training data to: {train_output_path}")
    print(f"Saving testing data to: {test_output_path}")

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    main()
