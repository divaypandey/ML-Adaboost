#412 be weird
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as mp
from pylab import show
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import cross_val_score
from sklearn import preprocessing # import normalize
from sklearn.preprocessing import normalize

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
#from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
import pandas as pd


#data = np.loadtxt("iris.data")
#from sklearn.datasets import load_iris

#iris = load_iris()
#type(iris) 
#print (iris.data)
#iris.data.shape

#print (iris.feature_names)
#print (iris.target)
#print (iris.target_names)

####

data = arff.loadarff('3year.arff')
df = pd.DataFrame(data[0])
df.replace(np.nan, 0, inplace=True)
df2= pd.DataFrame(data[0])

#df.head()
#df.drop("class", axis=1, inplace=True)
#print(df)
#print(df.values)

####

#iris = datasets.load_iris()
X = []#iris.data
y = []#iris.target

#print(iris.data)
#print(iris.target)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

for row in df.values:
    y.append(float(row[-1]))
    X.append(row[:-1])

#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

## TEST TRAIN END, ADA BEGIN

lNodes = [1,5,10,100,1000,10000]
AccAvgs = []
errorAAvgs = []
for node in lNodes:
    estimator = DecisionTreeClassifier(max_depth = 1)
#    # Create adaboost classifer object
    abc = AdaBoostClassifier(base_estimator=estimator,n_estimators=node)
#    # Train Adaboost Classifer
    modelA = abc.fit(X_train, y_train)
#    #Predict the response for test dataset
    y_pred = modelA.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



y_pred = []
############################################################################################################ADA END,  MULTICLASS BEGIN

standardizer = StandardScaler()
# Standardize features 
#features_standardized = standardizer.fit_transform(X_train)
# Train a radius neighbors classifier 
logistic_regression = LogisticRegression(random_state=0, multi_class="ovr",max_iter=1000)
# Train model 
model = logistic_regression.fit(X_train, y_train)

# Create two observations 
#new_observations = [[ 1,  1,  1,  1]]
# Predict the class of two observations 
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

######################################################## MULTICLASS END



#    #adaboost = AdaBoostClassifier(base_estimator=estimator,n_estimators=node)
#    #modelA = adaboost.fit(normVal,trainDigits)
#    errorA = cross_val_score(modelA,X_train,y_train,cv = 10)
#    errorsinA = [1-x for x in errorA]
#    AvgError = np.mean(errorsinA)
#    errorAAvgs.append(AvgError)
#    AccAvg = np.mean(errorA)
#    AccAvgs.append(AccAvg)
#    print("Adaboost depth 1 for %d" %node ,errorA)
    


#mp.plot(lNodes,AccAvgs,'o')
#mp.title("Figure 1 - Accuracy")
#mp.xlabel("# Nodes")
#mp.ylabel("Accuracy")
#mp.xscale('log')
#show()


#print()
#print("testing for depth 1")

## Model Accuracy, how often is the classifier correct?

#lNodes = [1,5,10,100,1000,10000]
#AccAvgs = []
#errorAAvgs = []
#for node in lNodes:
#    estimator = DecisionTreeClassifier(max_depth = 2)
#    # Create adaboost classifer object
#    abc = AdaBoostClassifier(base_estimator=estimator,n_estimators=node)
#    # Train Adaboost Classifer
#    modelA = abc.fit(X_train, y_train)
#    #Predict the response for test dataset
#    y_pred = modelA.predict(X_test)
#    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#    #adaboost = AdaBoostClassifier(base_estimator=estimator,n_estimators=node)
#    #modelA = adaboost.fit(normVal,trainDigits)
#    errorA = cross_val_score(modelA,X_train,y_train,cv = 10)
#    errorsinA = [1-x for x in errorA]
#    AvgError = np.mean(errorsinA)
#    errorAAvgs.append(AvgError)
#    AccAvg = np.mean(errorA)
#    AccAvgs.append(AccAvg)
#    print("Adaboost depth 1 for %d" %node ,errorA)
    


#mp.plot(lNodes,AccAvgs,'o')
#mp.title("Figure 1 - Accuracy")
#mp.xlabel("# Nodes")
#mp.ylabel("Accuracy")
#mp.xscale('log')
#show()




#from sklearn.ensemble import AdaBoostClassifier

## Import Support Vector Classifier
#from sklearn.svm import SVC
##Import scikit-learn metrics module for accuracy calculation
#from sklearn import metrics
#svc=SVC(probability=True, kernel='linear')

## Create adaboost classifer object
#abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)

## Train Adaboost Classifer
#model = abc.fit(X_train, y_train)

##Predict the response for test dataset
#y_pred = model.predict(X_test)

#print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


######################################### test for lowest error ###############
#estimator = DecisionTreeClassifier(max_depth = 1)
#adaboost = AdaBoostClassifier(base_estimator=estimator,n_estimators=1000)
#modelA = adaboost.fit(normVal,trainDigits)
#errorA = cross_val_score(modelA,normVal,trainDigits,cv = 10)
#errorsinA = [1-x for x in errorA]
#AvgError = np.mean(errorsinA)
#print("Adaboost depth 1, est: 1000 error:" ,AvgError)

#estimator = DecisionTreeClassifier(max_depth = 10)
#adaboost = AdaBoostClassifier(base_estimator=estimator,n_estimators=100)
#modelA = adaboost.fit(normVal,trainDigits)
#errorA = cross_val_score(modelA,normVal,trainDigits,cv = 10)
#errorsinA = [1-x for x in errorA]
#AvgError = np.mean(errorsinA)
#print("Adaboost depth 10, est: 100 error:" ,AvgError)


#estimator = DecisionTreeClassifier(max_depth = 1000)
#adaboost = AdaBoostClassifier(base_estimator=estimator,n_estimators=100)
#modelA = adaboost.fit(normVal,trainDigits)
#errorA = cross_val_score(modelA,normVal,trainDigits,cv = 10)
#errorsinA = [1-x for x in errorA]
#AvgError = np.mean(errorsinA)
#print("Adaboost depth 1000, est: 100 error:" ,AvgError)



############################# predict on lowest error ###########################
#estimator = DecisionTreeClassifier(max_depth = 1)
#adaboost = AdaBoostClassifier(base_estimator=estimator,n_estimators=1000)
#adaboost.fit(normVal,trainDigits)

#xPred = []
#yPred = []
#cPred = []

###################################### official graph but takes forever to run #####3
#for xP in range(-100,100):
#    xP = xP/100.0
#    for yP in range(-100,100):
#        yP = yP/100.0
#        xPred.append(xP)
#        yPred.append(yP)
#        print('currently predicting', xP, yP)
#        if(adaboost.predict([[xP,yP]])=="1.0"):
#            cPred.append("r")
#        else:
#            cPred.append("b")
#mp.scatter(X1,Y1,s=3, c=colors)
#mp.scatter(xPred,yPred,s=3,c=cPred,alpha=.2)

#################   graph  #################
#mp.title("Figure 4.10 (for depth:1 # estimators: 1000)")
#mp.xlabel("Average")
#mp.ylabel("Varience")
#show()


#data = arff.loadarff('3year.arff')
#df = pd.DataFrame(data[0])
#df2= pd.DataFrame(data[0])

#df.head()
#df.drop("class", axis=1, inplace=True)
#print(df)
#print(df.values)