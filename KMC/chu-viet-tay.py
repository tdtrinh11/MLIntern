from __future__ import print_function
import numpy as np
from sklearn.datasets import fetch_mldata
import time

data_dir = './data' # path to my data folder
mnist = fetch_mldata('MNIST original', data_home = data_dir)
print("Shape of mnist data:", mnist.data.shape)