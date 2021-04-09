'''
Created on April , 2021
@author:
'''

## Import libraries in python
import argparse
import time
import json
import logging
import sys
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import importlib
from scipy.stats import randint, expon, uniform

import tensorflow as tf
import sklearn as sk
from sklearn import svm
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.metrics import mean_squared_error
from math import sqrt

from rp_creator import input_gen
from network import network_fit

# Ignore tf err log
pd.options.mode.chained_assignment = None  # default='warn'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(logging.ERROR)

# random seed predictable
seed = 0
random.seed(seed)
np.random.seed(seed)

current_dir = os.path.dirname(os.path.abspath(__file__))

## Dataset path
train_FD001_path = current_dir +'/cmapss/train_FD001.csv'
test_FD001_path = current_dir +'/cmapss/test_FD001.csv'
RUL_FD001_path = current_dir+'/cmapss/RUL_FD001.txt'
FD001_path = [train_FD001_path, test_FD001_path, RUL_FD001_path]

train_FD002_path = current_dir +'/cmapss/train_FD002.csv'
test_FD002_path = current_dir +'/cmapss/test_FD002.csv'
RUL_FD002_path = current_dir +'/cmapss/RUL_FD002.txt'
FD002_path = [train_FD002_path, test_FD002_path, RUL_FD002_path]

train_FD003_path = current_dir +'/cmapss/train_FD003.csv'
test_FD003_path = current_dir +'/cmapss/test_FD003.csv'
RUL_FD003_path = current_dir +'/cmapss/RUL_FD003.txt'
FD003_path = [train_FD003_path, test_FD003_path, RUL_FD003_path]

train_FD004_path =current_dir +'/cmapss/train_FD004.csv'
test_FD004_path = current_dir +'/cmapss/test_FD004.csv'
RUL_FD004_path = current_dir +'/cmapss/RUL_FD004.txt'
FD004_path = [train_FD004_path, test_FD004_path, RUL_FD004_path]

## Assign columns name
cols = ['unit_nr', 'cycles', 'os_1', 'os_2', 'os_3']
cols += ['sensor_{0:02d}'.format(s + 1) for s in range(26)]
col_rul = ['RUL_truth']

## Read csv file to pandas dataframe
FD_path = ["none", FD001_path, FD002_path, FD003_path, FD004_path]
dp_str = ["none", "FD001", "FD002", "FD003", "FD004"]

## temporary model path for NN
model_path = current_dir +'/temp_net.h5'


def concat_vec(train_samples, test_samples):
    '''
    concatenate vectors for MLPs (this is not to be used for the case of CNN which allows 2D input)
    :param train_samples:
    :param test_samples:
    :return:
    '''
    # flatten 2D rp to 1D vector and concatenate all vectors.
    if len(train_samples.shape)==4:
        train_vec_samples = train_samples.reshape(train_samples.shape[0],
                                                  train_samples.shape[1]*train_samples.shape[2]*train_samples.shape[3])
        test_vec_samples = test_samples.reshape(test_samples.shape[0],
                                                  test_samples.shape[1]*test_samples.shape[2]*test_samples.shape[3])

    # concatenate all the flattened vectors. ( if 2D rps are already flattened)
    elif len(train_samples.shape)==3:
        train_vec_samples = train_samples.reshape(train_samples.shape[0],
                                                  train_samples.shape[1]*train_samples.shape[2])
        test_vec_samples = test_samples.reshape(test_samples.shape[0],
                                                  test_samples.shape[1]*test_samples.shape[2])


    return train_vec_samples, test_vec_samples



def main():
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='RPs creator')
    parser.add_argument('-i', type=int, help='Input sources', required=True)
    parser.add_argument('-l', type=int, default=10, help='sequence length')
    parser.add_argument('--method', type=str, default='rps', help='data representation: rps')
    parser.add_argument('--thres_type', type=str, default=None, required=False,
                        help='threshold type for RPs: distance or point ')
    parser.add_argument('--thres_value', type=int, default=50, required=False,
                        help='percentage of maximum distance or black points for threshold')
    parser.add_argument('--flatten', type=str, default='no', help='flatten rps array.')
    parser.add_argument('--visualize', type=str, default='yes', help='visualize rps.')
    parser.add_argument('--n_hidden1', type=int, default=100, required=False,
                        help='number of neurons in the first hidden layer')
    parser.add_argument('--n_hidden2', type=int, default=10, required=False,
                        help='number of neurons in the second hidden layer')
    parser.add_argument('--epochs', type=int, default=1000, required=False, help='number epochs for network training')
    parser.add_argument('--batch', type=int, default=700, required=False, help='batch size of BPTT training')
    parser.add_argument('--verbose', type=int, default=2, required=False, help='Verbose TF training')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run model on cpu or cuda.')

    args = parser.parse_args()


    dp = FD_path[args.i]
    subdataset = dp_str[args.i]
    sequence_length = args.l
    thres_type = args.thres_type
    thres_value = args.thres_value
    device = args.device
    method = args.method
    n_hidden1 = args.n_hidden1
    n_hidden2 = args.n_hidden2
    epochs = args.epochs
    batch = args.batch
    verbose = args.verbose

    flatten = args.flatten
    if flatten == 'yes':
        flatten = True
    elif flatten == 'no':
        flatten = False

    visualize = args.visualize
    if visualize == 'yes':
        visualize = True
    elif visualize == 'no':
        visualize = False


    # Sensors not to be considered (those that do not disclose any pattern in their ts)
    sensor_drop = ['sensor_01', 'sensor_05', 'sensor_06', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']

    start = time.time()

    print("Dataset: ", subdataset)
    print("Seq_len: ", sequence_length)

    """ Creates a new instance of the training-validation task and computes the fitness of the current individual """
    data_class = input_gen(data_path_list=dp, sequence_length=sequence_length, sensor_drop= sensor_drop, visualize=visualize)

    if method == 'rps':
        train_samples, label_array_train, test_samples, label_array_test = data_class.rps(
            thres_type=thres_type,
            thres_percentage=thres_value,
            flatten=flatten,
            visualize=visualize)

    elif method == 'jrp': # please implement any method if needed
        pass


    print ("train_samples.shape: ", train_samples.shape) # shape = (samples, sensors, height, width)
    print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
    print ("test_samples.shape: ", test_samples.shape) # shape = (samples, sensors, height, width)
    print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)

    train_samples, test_samples = concat_vec(train_samples, test_samples)

    print ("train_samples.shape: ", train_samples.shape) # shape = (samples, sensors, height, width)
    print ("label_array_train.shape: ", label_array_train.shape) # shape = (samples, label)
    print ("test_samples.shape: ", test_samples.shape) # shape = (samples, sensors, height, width)
    print ("label_array_test.shape: ", label_array_test.shape) # shape = (samples, ground truth)

    mlps_net = network_fit(train_samples, label_array_train, test_samples, label_array_test,
                           model_path = model_path, n_hidden1=n_hidden1, n_hidden2=n_hidden2, verbose=verbose)

    trained_net = mlps_net.train_net(epochs=epochs, batch_size=batch)
    rms, score = mlps_net.test_net(trained_net)


    print(subdataset + " test RMSE: ", rms)
    print(subdataset + " test Score: ", score)

    end = time.time()
    print("Computing time: ", end - start)


if __name__ == '__main__':
    main()
