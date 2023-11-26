import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# read in data
data = pd.read_csv("possum.csv")

# get rid of unneeded stats about possums 
dataNew1 = data.drop(['case', 'site', 'Pop', 'sex'], axis=1)
# fill the NAs 
dataNew1.age = dataNew1.age.fillna(dataNew1.age.mean())
#print(dataNew)

# what we what to know is which possum body dimensions are most correlated with age?
target = dataNew1.age
# make input feature list without the actual output
inputFeatures = dataNew1.drop('age', axis='columns')
inputFeatures.footlgth = inputFeatures.footlgth.fillna(inputFeatures.footlgth.mean())
# split the data
X_train, X_test, y_train, y_test = train_test_split(inputFeatures, target, test_size=0.2)

# Linear regression with gradient descent
model = SGDRegressor()
# train the model
model.fit(X_train, y_train)
#model.fit(dataNew[['hdlngth', 'skullw','totlngth','taill','footlgth','earconch','eye','chest','belly']], target)

# --- STUFF FOR REPORT ---
print("\n-----Linear Regression w/ Gradient Descent-----")
# for training data
print("Training Data")
print("Mean Squared Error:", mean_squared_error(y_train, model.predict(X_train)))
print("Mean Absolute Error:", mean_absolute_error(y_train, model.predict(X_train)))
print("R2 Score:", r2_score(y_train, model.predict(X_train)))
# # for test data
print("\nTest Data")
print("Mean Squared Error:", mean_squared_error(y_test, model.predict(X_test)))
print("Mean Absolute Error:", mean_absolute_error(y_test, model.predict(X_test)))
print("R2 Score:", r2_score(y_test, model.predict(X_test)))

# print the w vector
print("\nParameter Vector: \n", model.coef_)
print("Parameter Vector With Intercept: ", model.intercept_)
print("\n")
# for OLS
# get rid of unneeded stats about passenger 
dataNew2 = data.drop(['case', 'site', 'Pop', 'sex'], axis=1)
# add 1's to be 1st col for x0
dataNew2.insert(0, column = "x0", value = "1")  

# fill the NAs in the age
dataNew2.age = dataNew2.age.fillna(dataNew2.age.mean())
# what we what to know is which possum body dimensions are most correlated with age?
target = dataNew2.age

# make input feature list without the actual output
inputFeatures = dataNew2.drop('age', axis='columns')
inputFeatures.footlgth = inputFeatures.footlgth.fillna(inputFeatures.footlgth.mean())
inputFeatures = np.float64(inputFeatures)
# split the data
X_train, X_test, y_train, y_test = train_test_split(inputFeatures, target, test_size=0.2)

# for getting the w vector
#w = np.linalg.inv(np.transpose(X_train)* (X_train)) * np.transpose(X_train) * y_train
transps = np.transpose(X_train)
inverse = np.linalg.inv(np.matmul(transps, X_train))
mult1 = np.matmul(inverse, transps)
w = np.matmul(mult1, y_train)

# output predictions 
outputPredictionTraining = np.matmul(X_train, w)
outputPredictionTest = np.matmul(X_test, w)

print("\n-----OLS-----")
# for training data
print("Training Data")
print("Mean Squared Error:", mean_squared_error(y_train, outputPredictionTraining))
print("Mean Absolute Error:", mean_absolute_error(y_train, outputPredictionTraining))
print("R2 Score:", r2_score(y_train, outputPredictionTraining))
# # for test data
print("\nTest Data")
print("Mean Squared Error:", mean_squared_error(y_test, outputPredictionTest))
print("Mean Absolute Error:", mean_absolute_error(y_test, outputPredictionTest))
print("R2 Score:", r2_score(y_test, outputPredictionTest))

# print the w vector
print("\nParameter Vector With Intercept: ", w)
# print("\nTraining Data\n", outputPredictionTraining)
# print("\nTest Data\n",outputPredictionTest)
 