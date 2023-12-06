
# framingham_heart_disease.csv

# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# importing dataset
data_set=pd.read_csv("framingham_heart_disease.csv")

# drop irrelevant columns
data_set=data_set.drop("education", axis = 1)
df=pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())

# to get information
print(data_set.info())


# to drop null values
print(df.isna().sum())
df.dropna(inplace=True)


# to drop duplicate values
print('no ofduplicates',df.duplicated().sum())
print(df.shape)
print(df)
print(df.info())



# splitting independent and dependent features
'''x = df.drop('TenYearCHD',axis=1)
y=df['TenYearCHD']'''
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
print(x)
print(y)

# model selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#feature Scaling
from sklearn.preprocessing import StandardScaler
scalar= StandardScaler()
x_train= scalar.fit_transform(x_train)
x_test= scalar.transform(x_test)

# Create a Logistic Regression model
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression()

# Train the model on the training data
logistic_model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = logistic_model.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df2.to_string())


# Evaluate the model

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# evaluate predictions
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))