import sys
import os
import numpy as np
import pandas as pd

train = pd.read_csv("fashion-mnist_train.csv")
test = pd.read_csv("fashion-mnist_test.csv")

train.append(test)

print (train.shape)