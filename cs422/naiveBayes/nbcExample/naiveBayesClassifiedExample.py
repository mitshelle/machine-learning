import numpy as np
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# read in data
data = pd.read_csv("Titanic-Dataset.csv")

# get rid of unneeded stats about passenger 
dataNew = data.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis=1)
# print(dataNew)

# what we what to know is if they survived
target = dataNew.Survived 

# make input feature list without the Survived
inputFeatures = dataNew.drop('Survived', axis='columns')

# make the sex into one-hot encoding so they are numbers
sexEncoding = pd.get_dummies(inputFeatures.Sex)

inputFeatures = pd.concat([inputFeatures, sexEncoding], axis='columns')
inputFeatures = inputFeatures.drop('Sex', axis='columns')


inputFeatures.Age = inputFeatures.Age.fillna(inputFeatures.Age.mean())

X_train, X_test, y_train, y_test = train_test_split(inputFeatures, target, test_size=0.2)

#print(inputFeatures)

model = GaussianNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(y_test[:10])
print(model.predict(X_test[:10]))
