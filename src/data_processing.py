import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def load_data():
    """Load and return the iris dataset as pandas DataFrame"""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    data['species'] = data['target'].apply(lambda x: iris.target_names[x])
    return data

def preprocess_data(test_size=0.2, random_state=42):
    # Load data
    df = load_data()
    
    # Split into features and target
    X = df.drop(columns=['target', 'species'])
    y = df['target']
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Combine features and target for saving
    train_df = X_train.copy()
    train_df['target'] = y_train
    
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    return train_df, test_df


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

     # Save raw data
    raw_df = load_data()
    raw_df.to_csv('data/raw/iris.csv', index=False)
    
    # Process and save data
    train, test = preprocess_data()
    train.to_csv('data/processed/train.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)