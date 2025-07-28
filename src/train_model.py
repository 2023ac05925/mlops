import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def load_dataset(path):
    """Load dataset from CSV"""
    df = pd.read_csv(path)
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

def train_model():
    # Load data
    X_train, y_train = load_dataset('data/processed/train.csv')
    X_test, y_test = load_dataset('data/processed/test.csv')
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/iris_rf_model.joblib')

if __name__ == "__main__":
    train_model()