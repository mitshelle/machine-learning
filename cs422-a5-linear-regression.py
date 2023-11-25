import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# read in data
data = pd.read_csv("possum.csv")

# get rid of unneeded stats about passenger 
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
#print(np.hstack((model.intercept_[:,None], model.coef_)))
# sensitivity = model.recall_score()
# get accuracy 
# print(y_test[:10])
# print(model.predict(X_test[:10]))

# %%
# Create confusion matrix
# confusion = confusion_matrix(y_test, model.predict(X_test))
# sns.heatmap(confusion, annot=True, fmt="d", cmap="GnBu")
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()
# %%

# for OLS
 aa.insert(2, column = "Department", value = "B.Sc")  
# aa.head() 