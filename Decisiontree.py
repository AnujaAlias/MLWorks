
# Iris data

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing dataset
data_set=pd.read_csv("Iris1.csv")
df=pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())

# to get information
data_set.info()

# to drop null values
data_set.isna().sum()

# to drop duplicate values
print('no ofduplicates',data_set.duplicated().sum())
print(data_set.shape)

# data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(data_set, hue="Species")
plt.show()

# Extracting Independent and dependent Variable
x= data_set.iloc[:,1:-1].values
y= data_set.iloc[:,-1].values

# Importing the LabelEncoder
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

# feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)

# Fitting Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# Predicting the test set result
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df2.to_string())

# evaluate model
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



