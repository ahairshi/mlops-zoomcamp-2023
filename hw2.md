## Homework

The goal of this homework is to get familiar with tools like MLflow for experiment tracking and 
model management.


## Q1. Install the package

To get started with MLflow you'll need to install the appropriate Python package.

For this we recommend creating a separate Python environment, for example, you can use [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs), 
and then install the package there with `pip` or `conda`.

Once you installed the package, run the command `mlflow --version` and check the output.

What's the version that you have?


```python
!mlflow --version
```

    mlflow, version 2.3.2


## Q2. Download and preprocess the data

We'll use the Green Taxi Trip Records dataset to predict the amount of tips for each trip. 

Download the data for January, February and March 2022 in parquet format from [here](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

Use the script `preprocess_data.py` located in the folder [`homework`](homework) to preprocess the data.

The script will:

* load the data from the folder `<TAXI_DATA_FOLDER>` (the folder where you have downloaded the data),
* fit a `DictVectorizer` on the training set (January 2022 data),
* save the preprocessed datasets and the `DictVectorizer` to disk.

Your task is to download the datasets and then execute this command:

```
python preprocess_data.py --raw_data_path <TAXI_DATA_FOLDER> --dest_path ./output
```

Tip: go to `02-experiment-tracking/homework/` folder before executing the command and change the value of `<TAXI_DATA_FOLDER>` to the location where you saved the data.

So what's the size of the saved `DictVectorizer` file?

* 54 kB
* 154 kB
* 54 MB
* 154 MB


```python
%%capture downloads

! wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet -O ./data/green_tripdata_2022-01.parquet
! wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet -O ./data/green_tripdata_2022-02.parquet
! wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet -O ./data/green_tripdata_2022-03.parquet
! wget https://raw.githubusercontent.com/DataTalksClub/mlops-zoomcamp/main/cohorts/2023/02-experiment-tracking/homework/preprocess_data.py -O ./homework/preprocess_data.py
! wget https://raw.githubusercontent.com/DataTalksClub/mlops-zoomcamp/main/cohorts/2023/02-experiment-tracking/homework/hpo.py -O ./homework/hpo.py
! wget https://raw.githubusercontent.com/DataTalksClub/mlops-zoomcamp/main/cohorts/2023/02-experiment-tracking/homework/train.py -O ./homework/train.py
! wget https://raw.githubusercontent.com/DataTalksClub/mlops-zoomcamp/main/cohorts/2023/02-experiment-tracking/homework/register_model.py -O ./homework/register_model.py
```


```python
! python ./homework/preprocess_data.py --raw_data_path ./data --dest_path ./output
```


```python
! ls -lrth ./output/dv.pkl
```

    -rw-rw-r-- 1 ahairshi ahairshi 151K May 31 22:26 ./output/dv.pkl


## Q3. Train a model with autolog

We will train a `RandomForestRegressor` (from Scikit-Learn) on the taxi dataset.

We have prepared the training script `train.py` for this exercise, which can be also found in the folder `homework`. 

The script will:

* load the datasets produced by the previous step,
* train the model on the training set,
* calculate the RMSE score on the validation set.

Your task is to modify the script to enable **autologging** with MLflow, execute the script and then launch the MLflow UI to check that the experiment run was properly tracked. 

Tip 1: don't forget to wrap the training code with a `with mlflow.start_run():` statement as we showed in the videos.

Tip 2: don't modify the hyperparameters of the model to make sure that the training will finish quickly.

What is the value of the `max_depth` parameter:

* 4
* 6
* 8
* 10


```python
!python ./homework/train.py
```

    2023/05/31 22:27:23 INFO mlflow.tracking.fluent: Experiment with name 'nyc-taxi-experiment-hw' does not exist. Creating a new experiment.
    2023/05/31 22:27:24 INFO mlflow.tracking.fluent: Autologging successfully enabled for sklearn.
    2023/05/31 22:27:38 WARNING mlflow.utils.autologging_utils: MLflow autologging encountered a warning: "/home/ahairshi/anaconda3/lib/python3.10/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils."


<img width="947" alt="image" src="https://github.com/ahairshi/mlops-zoomcamp-2023/assets/1314201/4bcb71b5-7716-4f8a-9b12-d1d50a6bf3dd">

## Launch the tracking server locally for MLflow

Now we want to manage the entire lifecycle of our ML model. In this step, you'll need to launch a tracking server. This way we will also have access to the model registry. 

In case of MLflow, you need to:

* launch the tracking server on your local machine,
* select a SQLite db for the backend store and a folder called `artifacts` for the artifacts store.

You should keep the tracking server running to work on the next three exercises that use the server.

## Q4. Tune model hyperparameters

Now let's try to reduce the validation error by tuning the hyperparameters of the `RandomForestRegressor` using `optuna`. 
We have prepared the script `hpo.py` for this exercise. 

Your task is to modify the script `hpo.py` and make sure that the validation RMSE is logged to the tracking server for each run of the hyperparameter optimization (you will need to add a few lines of code to the `objective` function) and run the script without passing any parameters.

After that, open UI and explore the runs from the experiment called `random-forest-hyperopt` to answer the question below.

Note: Don't use autologging for this exercise.

The idea is to just log the information that you need to answer the question below, including:

* the list of hyperparameters that are passed to the `objective` function during the optimization,
* the RMSE obtained on the validation set (February 2022 data).

What's the best validation RMSE that you got?

* 1.85
* 2.15
* 2.45
* 2.85



```python
!pip install optuna --quiet
```


```python
!python ./homework/hpo.py
```

    2023/05/31 22:33:06 INFO mlflow.tracking.fluent: Experiment with name 'random-forest-hyperopt' does not exist. Creating a new experiment.
    [32m[I 2023-05-31 22:33:06,159][0m A new study created in memory with name: no-name-7d0335b5-dc0a-4ca2-984d-dd64cc78ed09[0m
    [32m[I 2023-05-31 22:33:07,928][0m Trial 0 finished with value: 2.451379690825458 and parameters: {'n_estimators': 25, 'max_depth': 20, 'min_samples_split': 8, 'min_samples_leaf': 3}. Best is trial 0 with value: 2.451379690825458.[0m
    [32m[I 2023-05-31 22:33:08,173][0m Trial 1 finished with value: 2.4667366020368333 and parameters: {'n_estimators': 16, 'max_depth': 4, 'min_samples_split': 2, 'min_samples_leaf': 4}. Best is trial 0 with value: 2.451379690825458.[0m
    [32m[I 2023-05-31 22:33:09,743][0m Trial 2 finished with value: 2.449827329704216 and parameters: {'n_estimators': 34, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 4}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:10,257][0m Trial 3 finished with value: 2.460983516558473 and parameters: {'n_estimators': 44, 'max_depth': 5, 'min_samples_split': 3, 'min_samples_leaf': 1}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:10,912][0m Trial 4 finished with value: 2.453877262701052 and parameters: {'n_estimators': 22, 'max_depth': 11, 'min_samples_split': 5, 'min_samples_leaf': 2}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:11,240][0m Trial 5 finished with value: 2.4720122094960733 and parameters: {'n_estimators': 35, 'max_depth': 3, 'min_samples_split': 4, 'min_samples_leaf': 2}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:12,619][0m Trial 6 finished with value: 2.4516421799356767 and parameters: {'n_estimators': 28, 'max_depth': 16, 'min_samples_split': 3, 'min_samples_leaf': 3}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:12,850][0m Trial 7 finished with value: 2.5374040268274087 and parameters: {'n_estimators': 34, 'max_depth': 1, 'min_samples_split': 7, 'min_samples_leaf': 1}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:13,656][0m Trial 8 finished with value: 2.455971238567075 and parameters: {'n_estimators': 12, 'max_depth': 19, 'min_samples_split': 10, 'min_samples_leaf': 4}. Best is trial 2 with value: 2.449827329704216.[0m
    [32m[I 2023-05-31 22:33:13,859][0m Trial 9 finished with value: 2.486106021576535 and parameters: {'n_estimators': 22, 'max_depth': 2, 'min_samples_split': 8, 'min_samples_leaf': 2}. Best is trial 2 with value: 2.449827329704216.[0m



## Q5. Promote the best model to the model registry

The results from the hyperparameter optimization are quite good. So, we can assume that we are ready to test some of these models in production. 
In this exercise, you'll promote the best model to the model registry. We have prepared a script called `register_model.py`, which will check the results from the previous step and select the top 5 runs. 
After that, it will calculate the RMSE of those models on the test set (March 2022 data) and save the results to a new experiment called `random-forest-best-models`.

Your task is to update the script `register_model.py` so that it selects the model with the lowest RMSE on the test set and registers it to the model registry.

Tips for MLflow:

* you can use the method `search_runs` from the `MlflowClient` to get the model with the lowest RMSE,
* to register the model you can use the method `mlflow.register_model` and you will need to pass the right `model_uri` in the form of a string that looks like this: `"runs:/<RUN_ID>/model"`, and the name of the model (make sure to choose a good one!).

What is the test RMSE of the best model?

* 1.885
* 2.185
* 2.555
* 2.955


```python
!python ./homework/register_model.py
```

    2023/05/31 22:53:38 WARNING mlflow.utils.autologging_utils: MLflow autologging encountered a warning: "/home/ahairshi/anaconda3/lib/python3.10/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils."
    Successfully registered model 'MidnightRandomForestWanderingRegressor'.
    2023/05/31 22:53:52 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation. Model name: MidnightRandomForestWanderingRegressor, version 1
    Created version '1' of model 'MidnightRandomForestWanderingRegressor'.


## Q6. Model metadata

Now explore your best model in the model registry using UI. What information does the model registry contain about each model?

* Version number
* Source experiment
* Model signature
* All the above answers are correct

## Submit the results

* Submit your results here: https://forms.gle/Fy1pvrPEKd4yjz3s6
* You can submit your solution multiple times. In this case, only the last submission will be used
* If your answer doesn't match options exactly, select the closest one


## Deadline

The deadline for submitting is 1 June 2023 (Thursday), 23:00 CEST (Berlin time). 

After that, the form will be closed.


```python

```
