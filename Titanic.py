# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from string import ascii_letters
import seaborn as sns            
# For visualizing missing values.
from scipy import stats 


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split


data = pd.read_csv("C:\Users\j\Documents\Notebooks\\train.csv") 
data2 =pd.read_csv("C:\Users\j\Documents\Notebooks\\test.csv") 
print(data.info())
combined=data.append(data2)

#combined.Age.fillna(value=0, inplace=True)
combined["Sex"]=pd.Categorical(combined.Sex).codes
combined["Embarked"]=pd.Categorical(combined.Embarked).codes
combined['FamilySize'] = combined['SibSp'] + combined['Parch']
combined['isAlone']=combined.FamilySize.apply(lambda x : x<1)
combined['NameLen'] = combined.Name.apply(lambda x : len(x))
combined['Title']=combined.Name.str.extract(r'([A-Za-z]+)\.')
combined['HasAge']=combined.Age/combined.Age

combined.HasAge.fillna(value=0,inplace=True)

combined.Cabin.fillna(value = 'X', inplace = True)

#'''Keep only the 1st character where Cabin is alphanumerical.'''
combined.Cabin = combined.Cabin.apply( lambda x : x[0])

#print(combined.Cabin.value_counts())


test=pd.get_dummies(combined['Title'])
test2=pd.get_dummies(combined['Cabin'])

#print(test.head())
#print("test")

combined=pd.concat([combined,test],axis=1)
combined=pd.concat([combined,test2],axis=1)

combined.info()
print(combined['HasAge'])
#combined.info()
#combined.head()
corr=combined.corr()

#sns.distplot(combined["Fare"].dropna(), hist=False)
            
print(corr)

#Dropping remaining non-numeric & irrelevant columns
combined=combined.drop(["Name"], axis=1)
combined=combined.drop(["Cabin"], axis=1)
combined=combined.drop(["Ticket"], axis=1)
combined=combined.drop(["Title"], axis=1)
combined.Age.fillna(value=20,inplace=True)
print("test")
print(combined.info())

reg=LogisticRegression()

d_train = combined.iloc[:890,:]
d_test = combined.iloc[890:, :]


X=d_train.drop(["Survived"], axis=1)
y=d_train.Survived

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=42)

reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
y_pred=np.greater(y_pred,.5)
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

# summarize the fit of the model
mse = np.mean((y_pred-y_test)**2)
#print reg.intercept_, reg.coef_, mse, 
print(reg.score(X, y))