import numpy as np
import matplotlib.pyplot as plt
import dataPreprocessing as dp

from tqdm import tqdm

from scipy.interpolate import BSpline, make_interp_spline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score

import scipy
import time

# numComponents = int(input('Enter number of components : '))

def getEigenVectors(data):
    cov_mat = np.cov(data.T)
    eigen_vals, eigen_vectors = scipy.linalg.eigh(cov_mat)
    # print ('Eigen Values:\n', eigen_vals)
    # print ('Eigen Vectors:\n', eigen_vectors)
    return eigen_vals, eigen_vectors
    
def getVarianceExplainedRatios(eigen_vals):
    tot = np.sum(eigen_vals)
    var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    
    # plot explained variances
    # print (len(eigen_vals))
    plt.bar(range(1,len(eigen_vals)+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1,len(eigen_vals)+1), cum_var_exp, where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.show()
    
    return 

def featureTransformation(eigen_vals, eigen_vectors, numComponents):
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vectors[:, i]) for i in range(len(eigen_vals))]
    eigen_pairs.sort(key = lambda k: k[0], reverse = True)

    # get the weight matrix now
    # W = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis], eigen_pairs[2][1][:, np.newaxis]))
    W = np.array(eigen_pairs[0][1][:, np.newaxis])
    if numComponents > 1:
        for i in range(1, numComponents):
            W = np.hstack((W, eigen_pairs[i][1][:, np.newaxis]))
            
    # print('Matrix W:\n', W)
    return W

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plotData(newX):
    plt.figure()
    plt.scatter(newX[:, 0], newX[:, 1])
    plt.xlabel('1st Principal Component')
    plt.ylabel('2nd Principal Component')
    plt.show()

def GaussianNaiveBayes(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    # print (y_pred, Y_test)
    cm = confusion_matrix(Y_test, y_pred)
    # print ('Accuracy of GNB: ', np.trace(cm)/np.sum(cm))
    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred, average='macro')
    return accuracy, precision 

def LogisticReg(X_train, X_test, Y_train, Y_test):
    clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial').fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    # print ('Accuracy of LogReg: ', np.trace(cm)/np.sum(cm))
    return np.trace(cm)/np.sum(cm)

def plotAccuracies(acc_g, prec_g):
    plt.figure()
    x = range(1, len(acc_g)+1)
    plt.plot(range(1, len(acc_g)+1), acc_g, label='Gaussian - Accuracy', color='green')
    plt.plot(range(1, len(prec_g)+1), prec_g, label='Gaussian - Precision', color='purple')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy of Model')
    plt.legend()
    plt.show()
    return  

def main():
    dataType = 3
    X, Y = dp.takeInputAuto(dataType)
    e_vals, e_vectors = getEigenVectors(X)
    # print (e_vals)
    # getVarianceExplainedRatios(e_vals)
    acc_gaussian = []
    prec_gaussian = []
    for i in tqdm(range(len(e_vals))):
        w = featureTransformation(e_vals, e_vectors, i)
        # Using this matrix W we will now multiply with the actual data, to get the new data
        newX_PCA = X @ w
        X_train, X_test, Y_train, Y_test = train_test_split(newX_PCA, Y, test_size=0.3)
        acc_gaussian.append(GaussianNaiveBayes(X_train, X_test, Y_train, Y_test)[0])
        prec_gaussian.append(GaussianNaiveBayes(X_train, X_test, Y_train, Y_test)[1])
    print ('Accuracy:\n', acc_gaussian)
    print ('Precision:\n', prec_gaussian)
    # np.save('acc_g_breast.npy', acc_gaussian)
    # np.save('prec_g_breast.npy', prec_gaussian)
    plotAccuracies(acc_gaussian, prec_gaussian)   

def newMain():
    for i in range(1, 5):
        dataset = ''
        optimalComp = 0
        if i == 1:
            dataset = 'breast'
        elif i == 2:
            dataset = 'heart'
        elif i == 3:
            dataset = 'sonar'
        
        print ('\n----------------------------------\n',dataset,'\n----------------------------------\n')
        
        X, Y = dp.takeInputAuto(i)
        e_vals, e_vectors = getEigenVectors(X)
        # print (e_vals)
        # getVarianceExplainedRatios(e_vals)
        '''
        acc_gaussian = []
        prec_gaussian = []
        for i in tqdm(range(len(e_vals))):
            w = featureTransformation(e_vals, e_vectors, i)
            # Using this matrix W we will now multiply with the actual data, to get the new data
            newX_PCA = X @ w
            X_train, X_test, Y_train, Y_test = train_test_split(newX_PCA, Y, test_size=0.3)
            acc_gaussian.append(GaussianNaiveBayes(X_train, X_test, Y_train, Y_test)[0])
            prec_gaussian.append(GaussianNaiveBayes(X_train, X_test, Y_train, Y_test)[1])
        # print ('Accuracy:\n', acc_gaussian)
        # print ('Precision:\n', prec_gaussian)
        plotAccuracies(acc_gaussian, prec_gaussian)
        optimalComp = np.argmax(np.add(acc_gaussian, prec_gaussian)) + 1
        print ('Accuracy New:', np.max(acc_gaussian))
        print ('Precision New:', np.max(prec_gaussian))
        print ('Optimal Component: ', optimalComp)

        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        # print (X_train.shape, X_train_.shape)

        acc = GaussianNaiveBayes(X_train, X_test, Y_train, Y_test)[0]
        prec = GaussianNaiveBayes(X_train, X_test, Y_train, Y_test)[1]
        
        print ('Accuracy:', acc)
        print ('Precision:', prec)
        '''
    
newMain()

