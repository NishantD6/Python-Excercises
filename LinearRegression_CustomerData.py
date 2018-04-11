#### Import the packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#### Import the Dataset
customers = pd.read_csv('/Users/Home/Desktop/Big_Data/Python/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Linear-Regression/Ecommerce Customers')

#### Explore the Datasets
customers.head()
customers.info()
customer.describe()

#### Visulize the Dataset
#Pair Plots
sns.pairplot(customers, size=1.6)

"""
As we can see, Yearly amount spend is correlated to:
1. Length of Membership (Strong Correlation),
2. Time on App (Week)
3. Average Session Length (Weak)
4. Time on Website (No Correlation)
"""

#Scatter Plots
sns.jointplot(customers['Time on Website'], customers['Yearly Amount Spent'])
sns.jointplot(customers['Time on App'], customers['Yearly Amount Spent'])
sns.jointplot(customers['Time on App'], customers['Yearly Amount Spent'],kind="hex")

#Scatter + Regression Plot
sns.regplot(customers['Length of Membership'], customers['Yearly Amount Spent'])


####Create Train Test Split
from sklearn.model_selection import train_test_split


#Quicky select the necessary columns
customers.columns

#Define the x (independent variables) & y(dependent variable)
x = customers[['Avg. Session Length',
               'Time on App',
               'Time on Website',
               'Length of Membership']]

y = customers[['Yearly Amount Spent']]

#Define training set & test set
x_train, x_test, y_train, ytest = train_test_split(x,y, test_size=0.3, random_state=101)

x_train.info()
x_test.info()

#Call the linear model and name it lm (or anyother name)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#Fit the lm model with training data
lm.fit(x_train, y_train)

###Evaulate Coefficients & Make Predictions on Test Dataset

#Coefficents of the different independent variables. As can be seen, length of membership is the highest correlation coefficient
lm.coef_
pd.DataFrame(lm.coef_.transpose(),x.columns, columns=['Coef'])

#Make predictions on the test data
predictions = lm.predict(x_test)

####Visualize predicted values to the real values
#Note, ytest is in pandas dataframe form and needs to be converted to numpy array form as predictions are in array form
plt.scatter(np.array(ytest), predictions)
sns.regplot(np.array(ytest), predictions)
sns.distplot(np.array(ytest)-predictions, bins=50)

#Or the array needs to be convereted into a Pandas DataFrame
plt.scatter(ytest, pd.DataFrame(predictions,columns=['Something']))
sns.regplot(ytest, pd.DataFrame(predictions,columns=['Something']))
sns.distplot(ytest-pd.DataFrame(predictions,columns=['Something']))

#### Determine Performance Metrics of the Linear Regression Model
from sklearn import metrics

print('MAE', metrics.mean_absolute_error(ytest,predictions))
print('MSE', metrics.mean_squared_error(ytest,predictions))
print('RSME', np.sqrt(metrics.mean_squared_error(ytest,predictions)))
print('R-Squared', metrics.explained_variance_score(ytest,predictions))
#Rsquare is how much variance the model explains. This R square value of .989 means that the model can explain about 99% of the variance.

