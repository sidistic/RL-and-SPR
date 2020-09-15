import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import KernelCenterer
from scipy.interpolate import BSpline, make_interp_spline
from sklearn.metrics import precision_score
from tqdm import tqdm



from scipy import exp
from scipy.linalg import eigh

from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
import dataPreprocessing as dp
import pandas as pd

from sklearn.datasets import make_moons

import timeit

def transform(gamma,X_test,X_train):

    K = np.square(euclidean_distances(X_test,X_train))
    K = exp(-gamma * K) 
    N1 = K.shape[0]
    one_n1 = np.ones((N1,N1)) / N1
    N2 = K.shape[1]
    one_n2 = np.ones((N2,N2)) / N2

    K = K - one_n1.dot(K) - K.dot(one_n2) + one_n1.dot(K).dot(one_n2) 

    return K



def rbf_kernel_pca(X, gamma, n_components):
    """
    RBF kernel PCA implementation.    
    Parameters
    ------------
    X: {NumPy ndarray}, shape = [n_examples, n_features]  
    gamma: float
        Tuning parameter of the RBF kernel    
    n_components: int
        Number of principal components to return    
    Returns
    ------------
    X_pc: {NumPy ndarray}, shape = [n_examples, k_features]
        Projected dataset   
    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')    
    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)    
    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)    
    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)    
    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order, where the eigenvectors are normalized
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    eigvecs = eigvecs[:, eigvals > 0]
    eigvals = eigvals[eigvals > 0]

    values = np.sqrt(np.sum(np.square(eigvecs),axis=1))
    tot = sum(eigvals)
    var_exp = [(i / tot) for i in sorted(eigvals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # plot explained variances
    plt.bar(range(1,len(eigvals)+1), var_exp, alpha=0.5,
            align='center', label='individual explained variance')
    plt.step(range(1,len(eigvals)+1), cum_var_exp, where='mid',
            label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.savefig('3-var.png')
    plt.show()

    return K, eigvals, eigvecs


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=[cmap(idx)],
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)# plot decision regions for training set

def GaussianNaiveBayes(X_train, X_test, Y_train, Y_test):
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, Y_train).predict(X_test)
    # print (y_pred, Y_test)
    cm = confusion_matrix(Y_test, y_pred)
    print ('Accuracy of GNB: ', np.trace(cm)/np.sum(cm))
    accuracy = gnb.score(X_test,Y_test)
    precision = precision_score(Y_test, y_pred, average='macro')
    return accuracy,precision

def LogisticReg(X_train, X_test, Y_train, Y_test):
    clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial').fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    # print ('Accuracy of LogReg: ', np.trace(cm)/np.sum(cm))
    return np.trace(cm)/np.sum(cm)

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plotAccuracies(acc_g, pre_g,name):
    plt.figure()
    x = range(1,len(acc_g)+1)
    a_BSpline_g = make_interp_spline(x, acc_g)
    a_BSpline_l = make_interp_spline(x, pre_g)

    acc_g_new = a_BSpline_g(x)
    acc_l_new = a_BSpline_l(x)
    
    ag = moving_average(acc_g, 10)
    al = moving_average(pre_g, 10)
    

    plt.plot(range(1, features), acc_g, label='Gaussian - Accuracy', color='green')
    plt.plot(range(1, features), pre_g, label='Gaussian - Precision', color='indigo')
    plt.xlabel('Number of Components')
    plt.ylabel('Accuracy of Model')
    plt.legend()
    plt.savefig(name)
    plt.show()
    return 

np.random.seed(0)

X, y = dp.takeInputAuto(4)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=0
)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

features = X_train.shape[1]

# print("sum",np.sum(X_train[1,:]))
kpca = KernelPCA(kernel="rbf", gamma=0.001, fit_inverse_transform=True,n_components=2)
X_kpca = kpca.fit_transform(X_train)

t_kpca = kpca.transform(X_test)

GaussianNaiveBayes(X_kpca, t_kpca, y_train, y_test)
LogisticRegression(X_kpca, t_kpca, y_train, y_test)

lr1 = LogisticRegression(multi_class='auto', solver='liblinear')
lr1.fit(X_kpca, y_train)
print(lr1.score(X_kpca,y_train))

plot_decision_regions(X_kpca, y_train, classifier=lr1)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()


print(lr1.score(t_kpca,y_test))

plot_decision_regions(t_kpca, y_test, classifier=lr1)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

gnb = GaussianNB()
gnb.fit(X_kpca, y_train)
# print("gg1",gnb.score(X_kpca,y_train))
# print("gg1",gnb.score(t_kpca,y_test))

################################################

K, _lambdas,_alphas = rbf_kernel_pca(X_train,gamma=0.001,n_components=2)


# print("Ksum",np.sum(K,axis=1))

X_pc = np.column_stack([_alphas[:, i]
                           for i in range(2)])  
rbf_kpca = K.dot(X_pc)

X_test1 = transform(0.001,X_test,X_train)

print("testsum",np.sum(X_test1[1,:]))

# _alphas = _alphas * np.sqrt(_lambdas)


X_pc = np.column_stack([_alphas[:, i]
                           for i in range(2)])

tt = _lambdas[:2]
tt1 = X_test1.dot(X_pc)
test_kpca = X_test1.dot(X_pc)

lr = LogisticRegression(multi_class='auto', solver='liblinear')
lr.fit(rbf_kpca, y_train)

print(lr.score(rbf_kpca,y_train))

plot_decision_regions(rbf_kpca, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()

print(lr.score(test_kpca,y_test))

plot_decision_regions(test_kpca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.savefig('3-2d.png')
plt.show()

gnb = GaussianNB()
gnb.fit(rbf_kpca, y_train)
print("gg",gnb.score(rbf_kpca,y_train))
print("gg",gnb.score(test_kpca,y_test))

acc_gaussian = []
acc_gaussian1 = []
pre_gaussian = []
pre_gaussian1 = []
acc_logreg = []
time = []
for i in tqdm(range(1, features)):
    start = timeit.default_timer()

    X_pc = np.column_stack([_alphas[:, j]
                           for j in range(i)]) 
    rbf_kpca = K.dot(X_pc)
    test_kpca = X_test1.dot(X_pc)
    
    accuracy,precision = GaussianNaiveBayes(rbf_kpca, test_kpca, y_train, y_test)

    # acc_logreg.append(LogisticReg(rbf_kpca, test_kpca, y_train, y_test))

    stop = timeit.default_timer()
    time.append(stop - start)

    acc_gaussian.append(accuracy)
    pre_gaussian.append(precision)


    kpca = KernelPCA(kernel="rbf", gamma=0.001, fit_inverse_transform=True,n_components=i)
    rbf_kpca1 = kpca.fit_transform(X_train)

    test_kpca1 = kpca.transform(X_test)

    accuracy,precision = GaussianNaiveBayes(rbf_kpca1, test_kpca1, y_train, y_test)
    acc_gaussian1.append(accuracy)
    pre_gaussian1.append(precision)

# np.save('acc_g_mnist1.npy', acc_gaussian)
# # np.save('acc_l_mnist.npy', acc_logreg)
# np.save('time1.npy', time)
# np.save('acc_ginbuild_mnist1.npy', acc_gaussian1)

plotAccuracies(acc_gaussian, pre_gaussian,"3-mine.png")
plotAccuracies(acc_gaussian1, pre_gaussian1,"3-inbuild.png")

print(np.max(acc_gaussian),np.argmax(acc_gaussian)+1,pre_gaussian[np.argmax(acc_gaussian)])

plt.plot(range(1, features),time,label="Time")
plt.xlabel('Number of Components')
plt.ylabel('time')
plt.legend(loc='lower right')
plt.savefig('3-time.png')
plt.show()
