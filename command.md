# Install dependencies
pip install -r requirements.txt

# Process data
python src/data_processing.py

## if DVC is not found
python -m pip show dvc | findstr "Location"
## if location return 
Location: C:\Users\SRI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages
## then add this toenvirinment variable
$env:Path += ";C:\Users\SRI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts"

## initialise dvc
python -m dvc init  ## dvc init -f  ## force initilize
python -m dvc add data/raw/iris.csv
mkdir ../dvc-storage  ## local dvc storage folder
python -m dvc remote add -d storage ../dvc-storage
python -m dvc push

# Track data with DVC
dvc commit -f  # Only needed if you modified data processing
dvc push       # Push data to remote storage

# Train model
python src/train_model.py

# Track model with DVC
dvc add models/iris_rf_model.joblib
git add models/.gitignore models/iris_rf_model.joblib.dvc

# Commit everything to Git
git add .
git commit -m "Initial implementation with Iris dataset"