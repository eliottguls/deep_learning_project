# Check the versions of libraries
import sys
print("Python : {}".format(sys.version))

import scipy
print("scipy : {}".format(scipy.__version__))

import numpy
print("numpy : {}".format(numpy.__version__))

import matplotlib
print("matplotlib : {}".format(matplotlib.__version__))

import pandas
print("pandas: {}".format(pandas.__version__))

import sklearn
print("sklearn : {}".format(sklearn.__version__))


# *** Step 1 - Load the data *** #
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
path = "iris.csv"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
dataset = read_csv(path, names=names)

# *** Step 2 - Summarize the dataset *** #
# 2.1 : Dimensions of Dataset
print("---- SHAPE ----")
print(dataset.shape)

# 2.2 : Peek at the Data
print("---- CHECK DATA ----")
print(dataset.head(20))

# 2.3 : Statistical summary
print("---- SUMMARY ----")
print(dataset.describe())

# 2.4 : Class distribution
print("---- DISTRIBUTION ----")
print(dataset.groupby('class').size())

# *** Step 3 - Data visualization *** #
# 3.1 : Univariate plots
dataset.hist()
pyplot.show()

# 3.2 : Multivariate plots
scatter_matrix(dataset)
pyplot.show()


# *** Step 4 - Evaluate some algorithms *** #
# 4.1 : Create a validation Dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)


# 4.2 : Build Models
models = []
# 4.2.1 - Logistic regression(linear) :
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

# 4.2.2 - Linear Discriminant analysis(linear)
models.append(('LDA', LinearDiscriminantAnalysis()))

# 4.2.3 - K-nearest neighbors(nonlinear)
models.append(('KNN', KNeighborsClassifier()))

# 4.2.4 - Classification and Regression Trees(nonlinear)
models.append(('CART', DecisionTreeClassifier()))

# 4.2.5 - Gaussian Naive Bayes(nonlinear)
models.append(('NB', GaussianNB()))

# 4.2.6 - Support Vector Machines(nonlinear)
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []

print("---- RESULTS ----")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s : %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    

# *** Step 5 - Compare algorithms *** #
pyplot.boxplot(results, labels=names)
pyplot.title("Algorithm comparison")
pyplot.show()

# *** Step 6 - Make predictions *** #
# Here SVM model was probably the most acurate model
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Evaluate Predictions
print("---- PREDICTIONS EVALUATE ----")
print("--> accuracy_score : \n", accuracy_score(Y_validation, predictions))
print("--> confusion_matrix : \n", confusion_matrix(Y_validation, predictions))
print("--> classification_report : \n", classification_report(Y_validation, predictions))




 













