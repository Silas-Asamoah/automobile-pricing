# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:29:36 2019

@author: C819934
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import collections 
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import mlflow
import mlflow.tensorflow
from absl import app
import shutil
import tempfile
from mlflow import pyfunc

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

        
COLUMN_TYPES = collections.OrderedDict([
    ("symboling", int),
    ("normalized-losses", float),
    ("make", str),
    ("fuel-type", str),
    ("aspiration", str),
    ("num-of-doors", str),
    ("body-style", str),
    ("drive-wheels", str),
    ("engine-location", str),
    ("wheel-base", float),
    ("length", float),
    ("width", float),
    ("height", float),
    ("curb-weight", float),
    ("engine-type", str),
    ("num-of-cylinders", str),
    ("engine-size", float),
    ("fuel-system", str),
    ("bore", float),
    ("stroke", float),
    ("compression-ratio", float),
    ("horsepower", float),
    ("peak-rpm", float),
    ("city-mpg", float),
    ("highway-mpg", float),
    ("price", float)
])
        

def raw_dataframe():
    path = tf.keras.utils.get_file(URL.split("/")[-1], URL)
    df = pd.read_csv(path,names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES, na_values="?")
    return df

def load_data(y_name="price", train_fraction=0.7, seed=None):
    data = raw_dataframe()
    data = data.dropna()
    np.random.seed(seed)
    
    #Split into train and test data
    X_train = data.sample(frac=train_fraction, random_state=seed)
    X_test = data.drop(X_train.index)
    y_train = X_train.pop(y_name)
    y_test = X_test.pop(y_name)
    return (X_train, y_train), (X_test,y_test)

def model_input_fn(features, labels, batch_size):
    shuffle = False
    shuffle_buffer_size = 10000
    if labels is not None:
        dataset = tf.data.Dataset.from_tensor_slices((dict(features),labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_size).repeat()
    else:
        dataset = dataset.batch(batch_size)
    return dataset

def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
        
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size is not None"
    dataset = dataset.batch(batch_size)
    return dataset

mlflow.tensorflow.autolog()

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100,type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')
parser.add_argument('--price_norm_factor', default=1000, type=float, help='price normalization factor')

def main(argv):
    with mlflow.start_run():
        args = parser.parse_args(argv[1:])
        (train_X, train_y),(test_X, test_y) = load_data()
        train_y /= args.price_norm_factor
        test_y /=args.price_norm_factor
        
        #train_input_fn = model_input_fn(train_X, train_y, args.batch_size)
        train_input_fn = lambda: model_input_fn(features = train_X, labels = train_y, batch_size = args.batch_size)
        test_input_fn = lambda: eval_input_fn(features = test_X, labels = test_y, batch_size = args.batch_size)
        
        feature_columns = [
                tf.feature_column.numeric_column(key="curb-weight"),
                tf.feature_column.numeric_column(key="highway-mpg")
        ]
        
        
        model = tf.estimator.LinearRegressor(feature_columns=feature_columns)
        model.train(input_fn=train_input_fn, steps=args.train_steps)
        
        eval_result = model.evaluate(input_fn=test_input_fn)
        average_loss = eval_result["average_loss"]
        
        print("\n" + 80* "*")
        print("\nRMS error for the test set: ${:.0f}".format(args.price_norm_factor * average_loss ** 0.5))
    
        input_dict = {
            "curb-weight": np.array([2000, 3000]),
            "highway-mpg": np.array([30,40])
            }
        #predict_input_fn = automobile_data.make_dataset(1, input_dict)
        predict_input_fn = lambda: model_input_fn(features =input_dict, labels = None, batch_size = args.batch_size)
        predict_results = model.predict(input_fn=predict_input_fn)
        print("\nPrediction results:")
        for i, prediction in enumerate(predict_results):
            msg = ("Curb weight: {: 4d}lbs, "
                   "Highway: {: 0d}mpg, "
                   "Prediction: ${: 9.2f}")
            msg = msg.format(input_dict["curb-weight"][i], input_dict["highway-mpg"][i], args.price_norm_factor * prediction["predictions"][0])
            print("   "+msg)
        print()
        
        car_specifications = {
                "curb-weight": tf.Variable([], dtype= tf.float64, name="curb-weight"),
                "highway-mpg": tf.Variable([], dtype=tf.float64, name="highway-mpg")
                }
        
        receiver_fn = tf.compat.v1.estimator.export.build_raw_serving_input_receiver_fn(car_specifications)
        temp = tempfile.mkdtemp()
        
        try:
            saved_estimator_path = model.export_saved_model(temp, receiver_fn).decode("utf-8")
            pyfunc_model = pyfunc.load_model(mlflow.get_artifact_uri(artifact_path = None))
#            tf.data.Dataset.from_tensor_slices(dict(predict))
            predict_data = tf.constant(value = np.array([[1000, 3000], [20, 30]]),
                                       dtype=tf.int32)
#            predict_data = tf.cast(predict_data, dtype=tf.Tensor.int32)
            test_data = pd.DataFrame(data=predict_data, columns=["curb-weight","highway-mpg"])
            predict_df = pyfunc_model.predict(test_data)
            
            #template = '\nOriginal prediction is "{}", reloaded prediction is "{}"'
            
#            for expec, pred in zip(predict_results, predict_df['classes']):
#                car_id = predict_df['classes'][predict_df.loc[predict_df['classes'] == pred].index[0]]
#                reloaded_label = print("The things")
        finally:
            shutil.rmtree(temp)
            
        
if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    app.run(main=main)
    
    
    
    