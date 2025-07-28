# src/model_training.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from mlflow.models.signature import infer_signature
import mlflow.pyfunc


def load_data():
    """Load processed data"""
    processed_dir = Path(__file__).resolve().parents[1] / 'data' / 'processed'
    train_df = pd.read_csv(processed_dir / 'train.csv')
    test_df = pd.read_csv(processed_dir / 'test.csv')
        
    # Make sure we're only using feature columns
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                   'petal length (cm)', 'petal width (cm)']
    
    X_train = train_df[feature_cols]
    y_train = train_df['species']
    X_test = test_df[feature_cols]
    y_test = test_df['species']

    
    return X_train, X_test, y_train, y_test

def train_and_register_best_model(X_train, X_test, y_train, y_test):
    mlflow.set_experiment("Iris_Classification")
    
    # Dictionary to store model performances
    model_performances = {}
    
    # Model 1: Logistic Regression
    with mlflow.start_run(run_name="Logistic_Regression") as lr_run:
        params = {
            'penalty': 'l2',
            'C': 1.0,
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        signature = infer_signature(X_train, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="logistic_regression",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        model_performances[lr_run.info.run_id] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model': model,
            'run_id': lr_run.info.run_id,
            'model_path': "logistic_regression"
        }

    # Model 2: Random Forest
    with mlflow.start_run(run_name="Random_Forest") as rf_run:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        signature = infer_signature(X_train, y_pred)
        
        mlflow.log_params(params)
        mlflow.log_metrics({
            'accuracy': accuracy,
            'f1_score': f1
        })
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest",
            signature=signature,
            input_example=X_train.iloc[:5]
        )
        
        model_performances[rf_run.info.run_id] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'model': model,
            'run_id': rf_run.info.run_id,
            'model_path': "random_forest"
        }

    # Select best model based on accuracy
    best_run_id = max(
        model_performances.keys(),
        key=lambda x: model_performances[x]['accuracy']
    )
    best_model_info = model_performances[best_run_id]
    
    print(f"\nBest model: {type(best_model_info['model']).__name__}")
    print(f"Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"F1 Score: {best_model_info['f1_score']:.4f}")

    # Register the best model
    model_name = "Iris_Classifier"
    model_version = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/{best_model_info['model_path']}",
        name=model_name
    )
    
    print(f"\nSuccessfully registered model '{model_name}'")
    print(f"Version: {model_version.version}")
    print(f"Run ID: {best_run_id}")

    # Test loading and prediction
    try:
        print("\nTesting loaded model...")
        loaded_model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version.version}"
        )
        
        # Use the same input example we logged with the model
        test_input = X_test.iloc[:5]
        predictions = loaded_model.predict(test_input)
        
        print("\nTest predictions from registered model:")
        print(f"Input:\n{test_input}")
        print(f"Predictions: {predictions}")
        
        return loaded_model, X_train.columns.tolist()  # Return model and feature names
        
    except Exception as e:
        print(f"\nError testing registered model: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Load data first
    X_train, X_test, y_train, y_test = load_data()
    
    # Train, register, and get the loaded model
    model, feature_names = train_and_register_best_model(X_train, X_test, y_train, y_test)
    
    # Example of how to use the returned model
    if model is not None:
        # Create proper new_data example with only feature columns
        new_data = pd.DataFrame([
            [5.1, 3.5, 1.4, 0.2],  # Example Iris setosa
            [6.7, 3.0, 5.2, 2.3],  # Example Iris virginica
            [5.9, 3.0, 4.2, 1.5]   # Example Iris versicolor
        ], columns=feature_names)  # Use feature columns only
        
        print("\nMaking predictions on new data:")
        print(f"New data:\n{new_data}")
        
        predictions = model.predict(new_data)
        print(f"Predictions: {predictions}")