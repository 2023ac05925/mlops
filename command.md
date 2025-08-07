# Install dependencies
pip install -r requirements.txt

# Process data
python src/data_processing.py

## if DVC is not found
python -m pip show dvc | findstr "Location"
## if location return 
Location: C:\Users\SRI\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages
## then add this to envirinment variable
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

###################
### Part 2  ###
###################

# Process data first
python src/data_preparation.py

# Then train models
python src/model_training.py

# open Mlflow Ui
Mlflow ui
## if  above comman does not works  use : python -m mlflow ui

##################
 ### Part 3 #####
###################
## 1. Build the Docker image:
docker build -t iris-api .


## 2. Run the container (make sure MLflow is running):
docker run -p 5001:5001 iris-api
## or
docker run -p 5001:5001 --add-host=host.docker.internal:host-gateway iris-api

## 3. tets API
$body = @{
    data = @(
        @{sepal_length=5.1; sepal_width=3.5; petal_length=1.4; petal_width=0.2},
        @{sepal_length=6.7; sepal_width=3.0; petal_length=5.2; petal_width=2.3}
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:5001/predict" -Method POST -ContentType "application/json" -Body $body

## test APT example 2
$body = @{
    data = @(
        @{
            sepal_length = 5.1
            sepal_width  = 3.5
            petal_length = 1.4
            petal_width  = 0.2
        }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:5001/predict" -Method POST -ContentType "application/json" -Body $body

# Stop all running containers
docker stop $(docker ps -q)
# or list all containers 
docker ps -a
# Stop it using either
docker stop <container_id_or_name>
# or force remove if needed
docker rm -f <container_id_or_name>

# Find the process ID
$pid = (Get-NetTCPConnection -LocalPort 5001).OwningProcess
# Stop the process
Stop-Process -Id $pid -Force

###########################
### Part 4 #########
########################

To generate your SSH key pair:
ssh-keygen -t rsa -b 4096 -C "2023ac05925@wilp.bits-pilani.ac.in"

## run LocalStack container

## Login to docker
Docker login
## after succesful login run below command
docker run -d -p 4566:4566 -p 4571:4571 localstack/localstack

## above command will start LocalStack and expose AWS services locally.
## LocalStack will emulate AWS services on your local machine.
## Use AWS CLI or SDKs with endpoint http://localhost:4566 to interact with LocalStack.


## to check aws is installed
AWS -- version 

aws config
## You will be prompted for:
## AWS Access Key ID: (for LocalStack, you can use any value: test
## AWS Secret Access Key: (for LocalStack, you can use any value: test
## Default region name: us-east-1
## Default output format: json

## For every AWS CLI command, add the --endpoint-url flag:
aws s3 ls --endpoint-url=http://localhost:4566

## To use LocalStack, always add --endpoint-url=http://localhost:4566 to your AWS CLI commands.
## To test, try creating a bucket in LocalStack:
aws s3 mb s3://mybucket --endpoint-url=http://localhost:4566
## show all S3 buckets you have created in LocalStack running on your machine.
aws s3 ls --endpoint-url=http://localhost:4566

## using LocalStack, you do not need to set SSH_KNOWN_HOSTS or EC2_IP for your local development and CI/CD with LocalStack.


###################
#### Part 5 ######
##################

pip install flask sqlite3


git add .
git commit -m "Test CI/CD pipeline"
git push