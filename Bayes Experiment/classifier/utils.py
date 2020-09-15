import numpy as np
import math
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import dirichlet

# Takes in one class data 
# input = [[][][]...]
# mu and var of dimension of Xi 

def Parameters_ccd(data):
    mu = np.mean(data, axis = 0)
    var = np.var(data,axis = 0)
    return mu,var

# input mean and variance of a xi
# Gives a scalar output (Gaussian Value)

def Gaussian_ccd(x,mean,var):
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

# Takes in one class data 
# input = [[][][]...]
# output - gives a tuple of class conditional probability values for xi given the input class 
def MNB(class_data):
    num_words = class_data.shape[1]
    cp = []
    alpha = 1
    total_words = class_data.sum()
    summed = class_data.sum(axis=0)
    for i in range(num_words):
        cp.append((summed[i] + alpha)/ (total_words + alpha*num_words +1))
    
    return np.array(cp)

def Accuracy(test_X, ClassConditionalForAll, Prior):
    Error = 0
    TotalTest = 0
    Final_Posterior = []
    for i in tqdm(test_X):
        for x in (test_X[i]):
            TotalTest+=1
            result, posteriors = BayesClassifiers(x,ClassConditionalForAll,Prior)
            Final_Posterior.append(posteriors)
            if(result != i):
                # print(result,"***",i)
                Error+=1
    return (TotalTest - Error)*100/TotalTest , Final_Posterior


def BayesClassifiers(X,ClassConditionalForAll, Prior):
    posteriors = {}
    for i in ClassConditionalForAll:
        posteriors[i] = getPosteriorsProbability(X, ClassConditionalForAll[i], Prior[i]) 
    return max(posteriors, key=posteriors.get), posteriors
    
def getPosteriorsProbability(X,prob,prior):
    p=float(1)
    for i in range(0,len(X)):
        p+= X[i] * np.log(prob[i])
    p+= np.log(prior)
    return p

def ReadClassesFile(filename):
    Y =[]
    ifile = open(filename, "r")
    for y in ifile:
        Y.append(y[:len(y)-1])
    ifile.close
    return Y 

# Y is the list of corresponding classes
def GetPriors(Y):
    freq = {}
    total_data_points = 0
    for i in Y:
        if i in freq :
            freq[i] += 1
        else:
            freq[i] = 1
        total_data_points +=1

    prior = {}
    for group in freq:
        prior[group] = float(freq[group])/float(total_data_points)
    
    return prior

def SeparateByClass(X,Y):
    n = 0
    grouped_dat ={}

    for i in tqdm(X):
        if Y[n] in grouped_dat:
            grouped_dat[Y[n]] = np.append(grouped_dat[Y[n]],[i],axis=0)
            # print(grouped_dat[Y[n]].shape)
            # print(grouped_dat[Y[n]])
            #print(temp)
        else :
            grouped_dat[Y[n]] = np.array([i])

        
        n+=1    
    #print(temp)
    return grouped_dat

def getProbabilityVectorForEachClass(class_sep_data):
    probVect = {}
    for i in class_sep_data:
        # print(i,class_sep_data[i])
        probVect[i] = MNB(class_sep_data[i])
    return probVect


def split_data_test_train(class_sep_data, Y):
    class_sep_test_data = {}
    class_sep_train_data = {}

    for i in class_sep_data:
        x_train, x_test, Ytrain, Ytest = train_test_split(class_sep_data[i], class_sep_data[i], test_size = 0.2)
        class_sep_train_data[i] = x_train
        class_sep_test_data[i] = x_test

    return class_sep_train_data, class_sep_test_data


def estimate_dirichlet_par(x_train): #input to dirichlet.mle N*K numpy array, N=train_samples, K=dimension_of_each_input
    par = {}
    alpha = 0.001
    for i in x_train:
        x_train[i]= x_train[i]*1.0
        x_train[i]+=alpha
        x = np.linalg.norm(x_train[i],axis=1,keepdims=True)
        x_train[i] = x_train[i] / x
        # x_train[i] = np.transpose(np.transpose(x_train[i])/(np.sum(x_train[i],axis=1)))

        par[i] = dirichlet.mle(x_train[i])
    return par
    
    
def predict_dirichlet(X,par,prior): #parameters should be a dict
    posteriors = {}
    p = float(1)
    alpha = 0.001
    X = X*1.0
    X+=alpha
    X = X/sum(X)
    for i in par:
        for j in range(0,len(par[i])):
            p+= par[i][j] * np.log(X[j])
        posteriors[i] = p * prior[i]
    return max(posteriors, key=posteriors.get)
        
def dirichlet_accuracy(X_test,par,prior): #X_test and par are dict
    total=0
    error=0
    for i in X_test:
        for val in X_test[i]:
            total+=1
            result = predict_dirichlet(val,par,prior)
            if (result != i):
                error+=1
    return (total - error)*100/total


def FindAlphaParameter(x_train):
    
    alphas = {}

    summation = 0
    for i in x_train:
        column = x_train[i].shape[1]
        mean = x_train[i].mean(axis=0)
        var = x_train[i].var(axis=0)

        summation+=np.log((mean*(1-mean)/var) - 1)
    
    summation = np.exp(summation/(column - 1))

    for i in x_train:
        mean = x_train[i].mean(axis=0)
        alphas[i] = mean*summation

    return alphas





