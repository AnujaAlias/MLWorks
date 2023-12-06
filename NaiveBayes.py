# spam

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
data=pd.read_csv("spam.csv")
print(data)
df=pd.DataFrame(data)
print(df)
print("Actual Dataset")
print(df.to_string())

# to get information
print(df.info())

# to drop null values
print(df.isna().sum())

# to drop duplicate values
print('no ofduplicates',df.duplicated().sum())
df=df.drop_duplicates()
print(df.shape)
print(df.info())

# replace ham and spam with numerical values 0 and 1
df["spam"] = df["Category"].replace({"ham":0,"spam":1})

# split into x and y
x= df.iloc[:,1]
y= df.iloc[:,-1]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

# Convert the text data into numerical features using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_count = v.fit_transform(x_train.values)
x_test_count = v.transform(x_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
n = MultinomialNB()
n.fit(x_train_count,y_train)
y_pred = n.predict(x_test_count)

# Predicting the Test set results
y_pred = n.predict(x_test_count)
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