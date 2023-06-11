## Weights & Biases workshop

* Video: https://www.youtube.com/watch?v=yNyqFMwEyL4
* Github repository: https://wandb.me/mlops-zoomcamp-github
* Slides: http://wandb.me/mlops-zoomcamp


## Homework with Weights & Biases

The goal of this homework is to get familiar with Weights & Biases for experiment tracking, model management, hyperparameter optimization, and many more.

Befor getting started with the homework, you need to have a Weights & Biases account. You can do so by visiting [wandb.ai/site](https://wandb.ai/site) and clicking on the **Sign Up** button.

# Q1. Install the Package

To get started with Weights & Biases you'll need to install the appropriate Python package.

For this we recommend creating a separate Python environment, for example, you can use [conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-envs), 
and then install the package there with `pip` or `conda`.

Following are the libraries you need to install:

* `pandas`
* `matplotlib`
* `scikit-learn`
* `pyarrow`
* `wandb`

Once you installed the package, run the command `wandb --version` and check the output.

What's the version that you have?

**Answer**

```python
! wandb --version
```

    wandb, version 0.15.4



# Q2. Download and preprocess the data

We'll use the Green Taxi Trip Records dataset to predict the amount of tips for each trip. 

Download the data for January, February and March 2022 in parquet format from [here](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

**Tip:** In case you're on [GitHub Codespaces](https://github.com/features/codespaces) or [gitpod.io](https://gitpod.io), you can open up the terminal and run the following commands to download the data:

```shell
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-03.parquet
```

Use the script `preprocess_data.py` located in the folder [`homework-wandb`](homework-wandb) to preprocess the data.

The script will:

* initialize a Weights & Biases run.
* load the data from the folder `<TAXI_DATA_FOLDER>` (the folder where you have downloaded the data),
* fit a `DictVectorizer` on the training set (January 2022 data),
* save the preprocessed datasets and the `DictVectorizer` to your Weights & Biases dashboard as an artifact of type `preprocessed_dataset`.

Your task is to download the datasets and then execute this command:

```bash
python preprocess_data.py \
  --wandb_project <WANDB_PROJECT_NAME> \
  --wandb_entity <WANDB_USERNAME> \
  --raw_data_path <TAXI_DATA_FOLDER> \
  --dest_path ./output
```

Tip: go to `02-experiment-tracking/homework-wandb/` folder before executing the command and change the value of `<WANDB_PROJECT_NAME>` to the name of your Weights & Biases project, `<WANDB_USERNAME>` to your Weights & Biases username, and `<TAXI_DATA_FOLDER>` to the location where you saved the data.

Once you navigate to the `Files` tab of your artifact on your Weights & Biases page, what's the size of the saved `DictVectorizer` file?

* 54 kB
* 154 kB
* 54 MB
* 154 MB

```python
! python preprocess_data.py \
  --wandb_project "mlops-zoomcamp-wandb-homework" \
  --wandb_entity "daisyfuentesahamed" \
  --raw_data_path "./data" \
  --dest_path ./output
```

    [34m[1mwandb[0m: Currently logged in as: [33mdaisyfuentesahamed[0m. Use [1m`wandb login --relogin`[0m to force relogin
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_021658-7ukmi2o5[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mradiant-hill-1[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/7ukmi2o5[0m
    [34m[1mwandb[0m: Adding directory to artifact (./output)... Done. 0.0s
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 🚀 View run [33mradiant-hill-1[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/7ukmi2o5[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 6 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_021658-7ukmi2o5/logs[0m
    
<img width="1118" alt="image" src="https://github.com/ahairshi/mlops-zoomcamp-2023/assets/1314201/6c92d3d8-2e94-4a48-b528-397a6f76c29e">


# Q3. Train a model with Weights & Biases logging

We will train a `RandomForestRegressor` (from Scikit-Learn) on the taxi dataset.

We have prepared the training script `train.py` for this exercise, which can be also found in the folder `homework-wandb`. 

The script will:

* initialize a Weights & Biases run.
* load the preprocessed datasets by fetching them from the Weights & Biases artifact previously created,
* train the model on the training set,
* calculate the MSE score on the validation set and log it to Weights & Biases,
* save the trained model and log it to Weights & Biases as a model artifact.

Your task is to modify the script to enable to add Weights & Biases logging, execute the script and then check the Weights & Biases run UI to check that the experiment run was properly tracked.

TODO 1: log `mse` to Weights & Biases under the key `"MSE"`

TODO 2: log `regressor.pkl` as an artifact of type `model`, refer to the [official docs](https://docs.wandb.ai/guides/artifacts) in order to know more about logging artifacts.

You can run the script using:

```bash
python train.py \
  --wandb_project <WANDB_PROJECT_NAME> \
  --wandb_entity <WANDB_USERNAME> \
  --data_artifact "<WANDB_USERNAME>/<WANDB_PROJECT_NAME>/NYC-Taxi:v0"
```

Tip 1: You can find the artifact address under the `Usage` tab in the respective artifact's page.

Tip 2: don't modify the hyperparameters of the model to make sure that the training will finish quickly.

Once you have successfully ran the script, navigate the `Overview` section of the run in the Weights & Biases UI and scroll down to the `Configs`. What is the value of the `max_depth` parameter:

* 4
* 6
* 8
* 10

```python
! python train.py \
  --wandb_project "mlops-zoomcamp-wandb-homework" \
  --wandb_entity "daisyfuentesahamed" \
  --data_artifact "daisyfuentesahamed/mlops-zoomcamp-wandb-homework/NYC-Taxi:v0"
```

    [34m[1mwandb[0m: Currently logged in as: [33mdaisyfuentesahamed[0m. Use [1m`wandb login --relogin`[0m to force relogin
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_022816-yderw39s[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mlemon-leaf-2[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/yderw39s[0m
    [34m[1mwandb[0m:   4 of 4 files downloaded.  
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run history:
    [34m[1mwandb[0m: MSE ▁
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run summary:
    [34m[1mwandb[0m: MSE 2.45398
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: 🚀 View run [33mlemon-leaf-2[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/yderw39s[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 3 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_022816-yderw39s/logs[0m
    
<img width="1330" alt="image" src="https://github.com/ahairshi/mlops-zoomcamp-2023/assets/1314201/2370e791-8356-46af-9a40-164bb7a544b5">


# Q4. Tune model hyperparameters

Now let's try to reduce the validation error by tuning the hyperparameters of the `RandomForestRegressor` using [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps). We have prepared the script `sweep.py` for this exercise in the `homework-wandb` directory.

Your task is to modify `sweep.py` to pass the parameters `n_estimators`, `min_samples_split` and `min_samples_leaf` from `config` to `RandomForestRegressor` inside the `run_train()` function. Then we will run the sweep to figure out not only the best best of hyperparameters for training our model, but also to analyze the most optimum trends in different hyperparameters. We can run the sweep using:

```bash
python sweep.py \
  --wandb_project <WANDB_PROJECT_NAME> \
  --wandb_entity <WANDB_USERNAME> \
  --data_artifact "<WANDB_USERNAME>/<WANDB_PROJECT_NAME>/NYC-Taxi:v0"
```

This command will run the sweep for 5 iterations using the **Bayesian Optimization and HyperBand** method proposed by the paper [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](https://arxiv.org/abs/1807.01774). You can take a look at the sweep on your Weights & Biases dashboard, take a look at the **Parameter Inportance Panel** and the **Parallel Coordinates Plot** to determine, and analyze which hyperparameter is the most important:

* `max_depth`
* `n_estimators`
* `min_samples_split`
* `min_samples_leaf`

```python
! python sweep.py \
  --wandb_project "mlops-zoomcamp-wandb-homework" \
  --wandb_entity "daisyfuentesahamed" \
  --data_artifact "daisyfuentesahamed/mlops-zoomcamp-wandb-homework/NYC-Taxi:v0"
```

    Create sweep with ID: vglc1gxu
    Sweep URL: https://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/sweeps/vglc1gxu
    [34m[1mwandb[0m: Agent Starting Run: ep28xstr with config:
    [34m[1mwandb[0m: 	max_depth: 19
    [34m[1mwandb[0m: 	min_samples_leaf: 4
    [34m[1mwandb[0m: 	min_samples_split: 3
    [34m[1mwandb[0m: 	n_estimators: 33
    [34m[1mwandb[0m: Currently logged in as: [33mdaisyfuentesahamed[0m. Use [1m`wandb login --relogin`[0m to force relogin
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_024255-ep28xstr[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mefficient-sweep-1[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🧹 View sweep at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/sweeps/vglc1gxu[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/ep28xstr[0m
    [34m[1mwandb[0m:   4 of 4 files downloaded.  
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run history:
    [34m[1mwandb[0m: MSE ▁
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run summary:
    [34m[1mwandb[0m: MSE 2.45075
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: 🚀 View run [33mefficient-sweep-1[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/ep28xstr[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_024255-ep28xstr/logs[0m
    [34m[1mwandb[0m: Sweep Agent: Waiting for job.
    [34m[1mwandb[0m: Job received.
    [34m[1mwandb[0m: Agent Starting Run: imz8o6as with config:
    [34m[1mwandb[0m: 	max_depth: 7
    [34m[1mwandb[0m: 	min_samples_leaf: 3
    [34m[1mwandb[0m: 	min_samples_split: 8
    [34m[1mwandb[0m: 	n_estimators: 31
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_024346-imz8o6as[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mdainty-sweep-2[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🧹 View sweep at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/sweeps/vglc1gxu[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/imz8o6as[0m
    [34m[1mwandb[0m:   4 of 4 files downloaded.  
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run history:
    [34m[1mwandb[0m: MSE ▁
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run summary:
    [34m[1mwandb[0m: MSE 2.45483
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: 🚀 View run [33mdainty-sweep-2[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/imz8o6as[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_024346-imz8o6as/logs[0m
    [34m[1mwandb[0m: Agent Starting Run: dy0oqfxk with config:
    [34m[1mwandb[0m: 	max_depth: 8
    [34m[1mwandb[0m: 	min_samples_leaf: 2
    [34m[1mwandb[0m: 	min_samples_split: 7
    [34m[1mwandb[0m: 	n_estimators: 14
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_024419-dy0oqfxk[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mtrue-sweep-3[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🧹 View sweep at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/sweeps/vglc1gxu[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/dy0oqfxk[0m
    [34m[1mwandb[0m:   4 of 4 files downloaded.  
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run history:
    [34m[1mwandb[0m: MSE ▁
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run summary:
    [34m[1mwandb[0m: MSE 2.45658
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: 🚀 View run [33mtrue-sweep-3[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/dy0oqfxk[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_024419-dy0oqfxk/logs[0m
    [34m[1mwandb[0m: Agent Starting Run: 085dksdg with config:
    [34m[1mwandb[0m: 	max_depth: 20
    [34m[1mwandb[0m: 	min_samples_leaf: 4
    [34m[1mwandb[0m: 	min_samples_split: 2
    [34m[1mwandb[0m: 	n_estimators: 34
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_024451-085dksdg[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mapricot-sweep-4[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🧹 View sweep at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/sweeps/vglc1gxu[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/085dksdg[0m
    [34m[1mwandb[0m:   4 of 4 files downloaded.  
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run history:
    [34m[1mwandb[0m: MSE ▁
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run summary:
    [34m[1mwandb[0m: MSE 2.4508
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: 🚀 View run [33mapricot-sweep-4[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/085dksdg[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_024451-085dksdg/logs[0m
    [34m[1mwandb[0m: Sweep Agent: Waiting for job.
    [34m[1mwandb[0m: Job received.
    [34m[1mwandb[0m: Agent Starting Run: f5yl29ty with config:
    [34m[1mwandb[0m: 	max_depth: 19
    [34m[1mwandb[0m: 	min_samples_leaf: 4
    [34m[1mwandb[0m: 	min_samples_split: 4
    [34m[1mwandb[0m: 	n_estimators: 46
    [34m[1mwandb[0m: Tracking run with wandb version 0.15.4
    [34m[1mwandb[0m: Run data is saved locally in [35m[1m/home/ahairshi/mlops-zoomcamp-wandb/wandb/run-20230611_024550-f5yl29ty[0m
    [34m[1mwandb[0m: Run [1m`wandb offline`[0m to turn off syncing.
    [34m[1mwandb[0m: Syncing run [33mtough-sweep-5[0m
    [34m[1mwandb[0m: ⭐️ View project at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework[0m
    [34m[1mwandb[0m: 🧹 View sweep at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/sweeps/vglc1gxu[0m
    [34m[1mwandb[0m: 🚀 View run at [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/f5yl29ty[0m
    [34m[1mwandb[0m:   4 of 4 files downloaded.  
    [34m[1mwandb[0m: Waiting for W&B process to finish... [32m(success).[0m
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run history:
    [34m[1mwandb[0m: MSE ▁
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: Run summary:
    [34m[1mwandb[0m: MSE 2.44983
    [34m[1mwandb[0m: 
    [34m[1mwandb[0m: 🚀 View run [33mtough-sweep-5[0m at: [34m[4mhttps://wandb.ai/daisyfuentesahamed/mlops-zoomcamp-wandb-homework/runs/f5yl29ty[0m
    [34m[1mwandb[0m: Synced 6 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
    [34m[1mwandb[0m: Find logs at: [35m[1m./wandb/run-20230611_024550-f5yl29ty/logs[0m



```python

```

# Q5. Link the best model to the model registry

Now that we have obtained the optimal set of hyperparameters and trained the best model, we can assume that we are ready to test some of these models in production. In this exercise, you'll create a model registry and link the best model from the Sweep to the model registry.

First, you will need to create a Registered Model to hold all the candidate models for your particular modeling task. You can refer to [this section](https://docs.wandb.ai/guides/models/walkthrough#1-create-a-new-registered-model) of the official docs to learn how to create a registered model using the Weights & Biases UI.

Once you have created the Registered Model successfully, you can navigate to the best run of your sweep, navigate to the model artifact created by the particular run, and click on the Link to Registry option from the UI. This would link the model artifact to the Registered Model. You can choose to add some suitable aliases for the Registered Model, such as `production`, `best`, etc.

Now that the model artifact is linked to the Registered Model, which of these information do we see on the Registered Model UI?

* Versioning
* Metadata
* Aliases
* Metric (MSE)
* Source run
* All of these
* None of these

**Answer to Q5**

`All of these`

## Submit the results

* Submit your results here: https://forms.gle/ndmTHeogFLeckSHm9
* You can submit your solution multiple times. In this case, only the last submission will be used
* If your answer doesn't match options exactly, select the closest one


## Deadline

The deadline for submitting is 6 June, 23:00 (Berlin time). 

After that, the form will be closed.
