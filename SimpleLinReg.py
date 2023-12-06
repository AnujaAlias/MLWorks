# student scores

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# load data
data_set= pd.read_csv('student_scores.csv')

# print data
print("Dataset")
df=pd.DataFrame(data_set)
print(df.to_string())

# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')
plt.title("Hours vs Percentage")
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

# Preparing data,  divide the data into "attributes" (inputs) and "labels" (outputs).
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values
print(x)
print(y)
print("X=\n",x)
df2=pd.DataFrame(x)
print("X Data-Hours")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y Data-Score")
print(df3.to_string())
print("Y array\n")
print(y)

# Splitting the dataset into training and test set .
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= .25, random_state=53)
from sklearn.linear_model import LinearRegression

# create an instance of linear regression
regressor= LinearRegression()
regressor.fit(x_train, y_train)
x_pred= regressor.predict(x_train)
print("Prediction result on Test Data")
y_pred = regressor.predict(x_test)
dfs=pd.DataFrame(x_test)
print("X-test")
print(dfs)
df2 = pd.DataFrame({'Actual Y-Data': y_test,
'Predicted Y-Data': y_pred})
print(df2.to_string())
plt.scatter(x_train, y_train, color="green")
plt.plot(x_train, x_pred, color="red")
plt.title("Score vs Hours (Training Dataset)")
plt.xlabel("Study Hours")
plt.ylabel("Student Score")
plt.show()

#visualizing the Test set results
plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, x_pred, color="red")
plt.title("Score vs Hours (Test Dataset)")
plt.xlabel("study Hours")
plt.ylabel(" Student Score")
plt.show()
print("Mean")
print(df['Scores'].mean())

# evaluate model
from sklearn import metrics
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
