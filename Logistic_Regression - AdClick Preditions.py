"""
Logistic Regression to predict whether a user will click an add.

"""

# Importing Packages
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

# Importing File
ad_data = pd.read_csv("/Users/Home/Desktop/Big_Data/Python/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Logistic-Regression/advertising.csv")

# Exploring the Data
ad_data.info()
ad_data.head(5)
ad_data.describe()

# Visualizing the Data

sns.heatmap(ad_data.isnull(), yticklabels=False , cbar=False)

sns.distplot(ad_data['Age'], bins=50)
sns.distplot(ad_data['Daily Time Spent on Site'], bins=50)
sns.distplot(ad_data['Area Income'], bins=50)
sns.distplot(ad_data['Daily Internet Usage'], bins=50)

sns.jointplot(ad_data['Daily Internet Usage'],ad_data['Daily Time Spent on Site'], color="green")
sns.jointplot(ad_data['Age'],ad_data['Area Income'], kind = "hex")
sns.jointplot(ad_data['Age'],ad_data['Daily Internet Usage'], kind="kde", color="maroon")

sns.pairplot(ad_data, size=1.4, hue="Clicked on Ad")


# Split the data into training set and testing set
x= ad_data[['Daily Time Spent on Site', 'Age', 'Daily Internet Usage', 'Male']]
y= ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

# Create the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
admodel = LogisticRegression()
admodel.fit(xtrain, ytrain)

# Predict on the model
predictions = admodel.predict(xtest)
ytest=np.array(ytest)

#Create Confustion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(confusion_matrix(ytest, predictions))
print(classification_report(ytest, predictions))
