import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

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


def takeInputAuto(typeDataset):
    # typeDataset = int(input('Enter\n1 : Breast Cancer\n2 : Heart Disease\n3 : Wine Data\n'))

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
        
        # print (X, Y)
        print (X.shape, Y.shape)
    elif typeDataset == 2:
        # Heart disease
        data = np.genfromtxt('heart-disease.data', delimiter=',')
        # print (data.shape)
        # separate X and Y
        X, Y = data[:, 0:13], data[:, 13]
        # print (X, Y)
        print (X.shape, Y.shape)
    else:
        # wine data
        data = np.genfromtxt('sonar.csv', delimiter=',')
        # print (data.shape)
        X, Y = data[:, 0:60], data[:, 60]
        # print (X, Y)
        print (X.shape, Y.shape)

    sc = StandardScaler() 
    X_final = sc.fit_transform(X)
    return X_final, Y

