import numpy as np
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# read in data
data = pd.read_csv("nba-players.csv")

# get rid of unneeded stats about passenger 
dataNew = data.drop(['name', 'id'], axis=1)
#print(dataNew)

# what we what to know is if they are gonna last in the league 
# for 5 years 
target = dataNew.target_5yrs 

# make input feature list without the actual output
inputFeatures = dataNew.drop('target_5yrs', axis='columns')
#print(inputFeatures)

# split the data
X_train, X_test, y_train, y_test = train_test_split(inputFeatures, target, test_size=0.2)

# Logistics Regression
# standardizes each feature big range of #s 
# normalization
#model = make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss"))
model = SGDClassifier(loss="log_loss")
# train the model
model.fit(X_train, y_train)


# --- STUFF FOR REPORT ---
# for training data
print("Training Data: ")
print(classification_report(y_train, model.predict(X_train)))
# log loss
print("Log Loss:")
print(log_loss(y_train, model.predict(X_train)))

# for test data
print("\nTest Data: ")
print(classification_report(y_test, model.predict(X_test)))

# log loss
print("Log Loss:")
print(log_loss(y_test, model.predict(X_test)))

# print the w vector
print("Parameter Vector:")
print(model.coef_)
print("Parameter Vector With Intercept:")
print(np.hstack((model.intercept_[:,None], model.coef_)))
# sensitivity = model.recall_score()
# get accuracy 
# print(y_test[:10])
# print(model.predict(X_test[:10]))

# %%
# Create confusion matrix
confusion = confusion_matrix(y_test, model.predict(X_test))
sns.heatmap(confusion, annot=True, fmt="d", cmap="GnBu")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%