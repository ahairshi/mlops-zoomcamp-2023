```python
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error
```

## Homework

The goal of this homework is to train a simple model for predicting the duration of a ride - similar to what we did in this module.


## Q1. Downloading the data

We'll use [the same NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
but instead of "**Green** Taxi Trip Records", we'll use "**Yellow** Taxi Trip Records".

Download the data for January and February 2022.

Read the data for January. How many columns are there?

* 16
* 17
* 18
* 19


```python
df = pd.read_parquet('./data/yellow_tripdata_2022-01.parquet')
```


```python
print(f"No of Columns : {len(df.columns.to_list())}")
```

    No of Columns : 19



```python
df.shape
```




    (2463931, 19)



## Q2. Computing duration

Now let's compute the `duration` variable. It should contain the duration of a ride in minutes. 

What's the standard deviation of the trips duration in January?

* 41.45
* 46.45
* 51.45
* 56.45


```python
df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
print(f"The standard deviation of the trips duration in January is {round(df.duration.std(),2)}")
```

    The standard deviation of the trips duration in January is 46.45



## Q3. Dropping outliers

Next, we need to check the distribution of the `duration` variable. There are some outliers. Let's remove them and keep only the records where the duration was between 1 and 60 minutes (inclusive).

What fraction of the records left after you dropped the outliers?

* 90%
* 92%
* 95%
* 98%


```python
df = df[(df.duration >= 1) & (df.duration <= 60)]
```


```python
df.shape
```




    (2421440, 20)




```python
print(f"Fraction of the records left after Dropping the outliers {round(2421440/ 2463931 * 100, 0)}")
```

    Fraction of the records left after Dropping the outliers 98.0


## Q4. One-hot encoding

Let's apply one-hot encoding to the pickup and dropoff location IDs. We'll use only these two features for our model. 

* Turn the dataframe into a list of dictionaries
* Fit a dictionary vectorizer 
* Get a feature matrix from it

What's the dimensionality of this matrix (number of columns)?

* 2
* 155
* 345
* 515
* 715


```python
categorical = ['PULocationID', 'DOLocationID']
def read_dataframe(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df
```


```python
df_train = read_dataframe('./data/yellow_tripdata_2022-01.parquet')
dv = DictVectorizer()

train_dicts = df_train[categorical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

print(f"The Dimensionality of Matrix :  {X_train.get_shape()[1]}")
```

    The Dimensionality of Matrix :  515



```python
len(dv.feature_names_)
```




    515



## Q5. Training a model

Now let's use the feature matrix from the previous step to train a model. 

* Train a plain linear regression model with default parameters 
* Calculate the RMSE of the model on the training data

What's the RMSE on train?

* 6.99
* 11.99
* 16.99
* 21.99


```python
target = 'duration'
y_train = df_train[target].values

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

print(f"The RMSE on train : {round(mean_squared_error(y_train, y_pred, squared=False),2)}")
```

    The RMSE on train : 6.99


## Q6. Evaluating the model

Now let's apply this model to the validation dataset (February 2022). 

What's the RMSE on validation?

* 7.79
* 12.79
* 17.79
* 22.79


```python
df_val = read_dataframe('./data/yellow_tripdata_2022-02.parquet')

val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts)
y_pred = lr.predict(X_val)
y_val = df_val.duration.values
print(f"The RMSE on validation : {round(mean_squared_error(y_val, y_pred, squared=False),2)}")
```

    The RMSE on validation : 7.79


## Submit the results

* Submit your results here: https://forms.gle/uYTnWrcsubi2gdGV7
* You can submit your solution multiple times. In this case, only the last submission will be used
* If your answer doesn't match options exactly, select the closest one


## Deadline

The deadline for submitting is 23 May 2023 (Tuesday), 23:00 CEST (Berlin time). 

After that, the form will be closed.
