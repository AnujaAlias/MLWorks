
# Breast Cancer

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
data=pd.read_csv("breast_cancer.csv")

# drop the extra unamed column
data=data.drop("Unnamed: 32",axis=1)
df=pd.DataFrame(data)
print(df)
print("Actual Dataset")
print(df.to_string())

# to get information
print(df.info())

# to drop null values
print(df.isnull().sum())

# to drop duplicate values
print('no ofduplicates',df.duplicated().sum())

# encoding categorical values to numerical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["diagnosis"]=label_encoder.fit_transform(df["diagnosis"])

# splitting into training and testing data
x= df.drop(['id', 'diagnosis'], axis=1)  # Drop unnecessary columns
y = df['diagnosis']

# model selection
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

# create random forest model
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")

# Train the model on the training data
classifier.fit(x_train, y_train)

#Predicting the test set result
y_pred= classifier.predict(x_test)
print("------------PREDICTION----------")
df2=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df2.to_string())

# evaluate model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# evaluate predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
