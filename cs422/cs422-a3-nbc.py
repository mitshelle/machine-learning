import numpy as np
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# read in data
data = pd.read_csv("nba-players.csv")

# get rid of unneeded stats about passenger 
dataNew = data.drop(['name', 'id'], axis=1)
print(dataNew)

# what we what to know is if they are gonna last in the league 
# for 5 years 
target = dataNew.target_5yrs 

# make input feature list without the Survived
inputFeatures = dataNew.drop('target_5yrs', axis='columns')

print(inputFeatures)

X_train, X_test, y_train, y_test = train_test_split(inputFeatures, target, test_size=0.2)

# #print(inputFeatures)

model = GaussianNB()
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

# print(y_test[:10])
# print(model.predict(X_test[:10]))
