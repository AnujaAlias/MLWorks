# diabetes prediction

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as mlt
# importing dataset
data=pd.read_csv("diabetes_prediction_dataset.csv")
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
# encoding categorical values to numerical
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["smoking_history"]=label_encoder.fit_transform(df["smoking_history"])
df["gender"]=label_encoder.fit_transform(df["gender"])

#Separate input features and output feature into x and y
x = df.drop('diabetes', axis=1)  # Features
y = df['diabetes']
 # Spliting the data into training and testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
# scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
# petform KNN
from sklearn.neighbors import KNeighborsClassifier
# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k)
# Train the classifier on the training set
knn_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred= knn_classifier.predict(x_test)

# Evaluate the model's performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
print("Classification report:", classification_report(y_test, y_pred))
