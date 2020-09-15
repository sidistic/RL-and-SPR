import numpy as np
import utils
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

X = np.load("data/x_data.npy")
Y = utils.ReadClassesFile("data/classes.txt")

dat_seperated_by_classes = utils.SeparateByClass(X,Y)

# temp_data = dat_seperated_by_classes['0']
# temp_data = np.append(temp_data,dat_seperated_by_classes['1'],axis=0)
# temp_data = np.append(temp_data,dat_seperated_by_classes['5'],axis=0)
# temp_data = np.append(temp_data,dat_seperated_by_classes['8'],axis=0)
# temp_data = np.append(temp_data,dat_seperated_by_classes['9'],axis=0)
# temp_data = np.append(temp_data,dat_seperated_by_classes['3'],axis=0)


# Y1 = ['0'] * dat_seperated_by_classes['0'].shape[0]
# Y1 = np.array(Y1)
# Y1 = np.append(Y1,['1'] * dat_seperated_by_classes['1'].shape[0],axis=0)
# Y1 = np.append(Y1,['5'] * dat_seperated_by_classes['5'].shape[0],axis=0)
# Y1 = np.append(Y1,['8'] * dat_seperated_by_classes['8'].shape[0],axis=0)
# Y1 = np.append(Y1,['9'] * dat_seperated_by_classes['9'].shape[0],axis=0)
# Y1 = np.append(Y1,['3'] * dat_seperated_by_classes['3'].shape[0],axis=0)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

print("In-Built MNB")
clf = MultinomialNB()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

train_data = utils.SeparateByClass(x_train, y_train)

test_data = utils.SeparateByClass(x_test, y_test)



Priors = utils.GetPriors(Y)

print("MNB")
prob_vect_by_classes = utils.getProbabilityVectorForEachClass(train_data)

ResultAccuracy, Final_Posterors = utils.Accuracy(test_data, prob_vect_by_classes, Priors)
print(ResultAccuracy)

#Dirichlet
print("Dirichlet")
AlphaParameters = utils.estimate_dirichlet_par(train_data)
ResultAccuracy = utils.dirichlet_accuracy(test_data,AlphaParameters,Priors)
print(ResultAccuracy)
#Dirichlet
print("Dirichlet1")
AlphaParameters1 = utils.FindAlphaParameter(train_data)
ResultAccuracy = utils.dirichlet_accuracy(test_data,AlphaParameters1,Priors)
print(ResultAccuracy)
