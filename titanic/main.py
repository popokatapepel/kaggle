# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import Imputer

train_df = pd.read_csv('data/train.csv')
predict_df = pd.read_csv('data/test.csv')



train_df['Sex'].replace(['female','male'],[0,1],inplace=True)
predict_df['Sex'].replace(['female','male'],[0,1],inplace=True)

#X = train_df.ix[:, ['Age', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']].values

X = train_df.ix[:, ['Age', 'Pclass', 'Sex', 'SibSp']].values
X_predict=predict_df.ix[:, ['Age', 'Pclass', 'Sex', 'SibSp']].values
#X = train_df.ix[:, ['Age', 'Pclass', 'Sex']].values

y=train_df['Survived'].values

print(X)
print(X_predict)



imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X)
X=imp.transform(X)
X_predict=imp.transform(X_predict)


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.4, random_state=42)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5


classifiers = [
    LogisticRegression(max_iter=1000)]





names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


mx=0
for name, clf in zip(names, classifiers):
	print('#'*50)
	print(clf)
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	if score>mx:
		mx=score
		mxname=name
		out=clf.predict(X_predict)
	print(name, score)
print('#'*50)
print(mxname,mx)

dfout=pd.DataFrame(data={'PassengerId': predict_df['PassengerId'].values, 'Survived': out})

dfout.to_csv('out.csv',index=False)







