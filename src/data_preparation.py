import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def prepare_data():
    # Define paths
    project_dir = Path(__file__).resolve().parents[1]
    raw_data_path = project_dir / 'data' / 'raw' / 'iris.csv'
    processed_dir = project_dir / 'data' / 'processed'
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    
    # Basic data cleaning (example - adjust as needed)
    df = df.dropna()  # Remove any missing values
    df = df.drop_duplicates()  # Remove duplicates
    
    # Split into features and target
    X = df.drop('species', axis=1)
    y = df['species']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Combine features and target for saving
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save processed data
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    
    print(f"Data preparation complete. Files saved to {processed_dir}")

if __name__ == "__main__":
    prepare_data()