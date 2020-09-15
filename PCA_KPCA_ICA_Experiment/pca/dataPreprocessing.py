import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from keras.datasets import mnist


from sklearn.preprocessing import StandardScaler

def takeInput():
    typeDataset = int(input('Enter\n1 : Breast Cancer\n2 : Heart Disease\n3 : Wine Data\n'))

    if typeDataset == 1:
        data = np.genfromtxt('wdbc.csv', delimiter=',')
        X = data[:, 2:]
        Y = data[:, 1]
        # for i in range(len(data)):
        #     if data[i][9] == 2.0:
        #         data[i][9] = 0.0
        #     else:
        #         data[i][9] = 1.0
        # separate X and Y
        
        print (X, Y)
        print (X.shape, Y.shape)
    elif typeDataset == 2:
        # Heart disease
        data = np.genfromtxt('heart-disease.data', delimiter=',')
        # print (data.shape)
        # separate X and Y
        X, Y = data[:, 0:13], data[:, 13]
        # print (X, Y)
        # print (X.shape, Y.shape)
    else:
        # wine data
        data = np.genfromtxt('wine.data', delimiter=',')
        # print (data.shape)
        X, Y = data[:, 1:14], data[:, 0]
        # print (X, Y)
        # print (X.shape, Y.shape)

    sc = StandardScaler() 
    X_final = sc.fit_transform(X)
    return X_final, Y

def getXTrainForDigit(digits):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    count = 0
    for i in range(len(y_train)):
        if y_train[i] in digits:
            count += 1
    
    XTrainDigit = np.zeros((count, 28, 28))
    YTrainDigit = np.zeros(count)
    j = 0
    for i in range(len(y_train)):
        if y_train[i] in digits:
            YTrainDigit[j] = y_train[i]
            XTrainDigit[j] = x_train[i]
            j += 1
    
    XTrainDigit = np.reshape(XTrainDigit, (count, 784))
    print (XTrainDigit.shape, YTrainDigit.shape)
    # print (YTrainDigit)
    return XTrainDigit, YTrainDigit

def takeInputAuto(typeDataset):
    # typeDataset = int(input('Enter\n1 : Breast Cancer\n2 : Heart Disease\n3 : Wine Data\n'))

    if typeDataset == 1:
        data = np.genfromtxt('wdbc.csv', delimiter=',')
        X = data[:, 2:]
        Y = data[:, 1]
        print (X.shape)
    elif typeDataset == 2:
        # Heart disease
        data = np.genfromtxt('heart-disease.data', delimiter=',')
        # print (data.shape)
        # separate X and Y
        X, Y = data[:, 0:13], data[:, 13]
        # print (X, Y)
        print (X.shape)
    elif typeDataset == 3:
        # wine data
        data = np.genfromtxt('sonar.csv', delimiter=',')
        # print (data.shape)
        X, Y = data[:, 0:60], data[:, 60]
        # print (X, Y)
        print (X.shape)
    else:
        X, Y = getXTrainForDigit([1, 8])

    sc = StandardScaler() 
    X_final = sc.fit_transform(X)
    return X_final, Y

