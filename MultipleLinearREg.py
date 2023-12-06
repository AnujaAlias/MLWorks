# house dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load dataset
data= pd.read_csv('economic_index.csv')
print(data)
df=pd.DataFrame(data)
print(df)

# splitting dependent and independent variables
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
df2=pd.DataFrame(x)
print("X=")
print(df2)
df3=pd.DataFrame(y)
print("Y=")
print(df3)

# model selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)


# create multiple liner regression model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()

# Train the model on the training data and Make predictions on the test data
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)
y_pred
y_test
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())
print("Mean")
print(df.describe())

# evaluate model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")

# Plotting the predicted prices against the actual prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs. Predicted Prices')
plt.show()