#### Import Packages & Data sets

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

train = pd.read_csv('/Users/Home/Desktop/Big_Data/Python/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Logistic-Regression/titanic_train.csv')
test = pd.read_csv('/Users/Home/Desktop/Big_Data/Python/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Logistic-Regression/titanic_test.csv')

#### Explore the Train Dataset
train.info()
train.head()
train.describe()

#### HeatMap to create the missing data - Yellow is missing info
sns.heatmap(train.isnull(), yticklabels=False,cbar=False, cmap='viridis')

# We can replace missing age data as it is only 20% - dropping this rows would waste a lot of good data.
# We can remove the whole cabin column.

#### Explore the Data

#How many survived vs. did-not-survive
sns.countplot(x='Survived', data=train)
sns.countplot(x='Survived', hue='Sex', data=train) #By Gender
sns.countplot(x='Survived', hue='Pclass', data=train) #By Passenger Class
sns.factorplot(x='Pclass',hue='Sex', col='Survived', data=train, kind="count") #By Class & Gender

#How many survived vs. did-not-survive (As percentages)
ax=sns.barplot(x='Survived', y='Survived', data=train, estimator=lambda Survived : len(Survived)/len(train)*100)
ax.set(ylabel="Percent")

ax= sns.barplot(x='Survived', y='Survived', hue='Sex', estimator=lambda Survived : len(Survived)/len(train)*100, data=train) #By Gender
ax.set(ylabel="Percent")

ax= sns.barplot(x='Survived', y='Survived', hue='Pclass', data=train, estimator= lambda Survived : len(Survived)/len(train)*100)
ax.set(ylabel='Percent')

#Age Distribution by Graph
sns.distplot(train['Age'].dropna(), bins=40)
train.groupby('Sex').Age.plot(kind='kde', legend=True) #density plot by Gender
train.groupby('Sex').Age.hist(alpha=0.3)
train.groupby('Pclass').Age.plot(kind='density', legend=True) #density plot by Class

#Age Distribution by Averages
train.groupby('Sex').Age.mean().dropna(how='any')
train.groupby('Pclass').Age.mean().dropna(how='any')
train.groupby(['Sex','Pclass']).Age.mean().dropna(how='any')

"""
So we can see that Pclass and Sex do change with the Age. 
We will fill in the missing age based on the averages. 

"""
#### Cleaning the Data

#Filling Missing Age based on Gender & Pclass
train.describe()
def imputey(cols):
    Age = cols[0]
    Sex = cols[1]
    Pclass = cols[2]

    if pd.isnull(Age):
        if Sex == "female":
            if Pclass == 1:
                return 34.6
            elif Pclass == 2:
                return 28.7
            else:
                return 21.75
        else:
            if Pclass == 1:
                return 41.28
            elif Pclass == 2:
                return 30.74
            else:
                return 26.5
    else:
        return Age

train['Age'] = train[['Age', 'Sex', 'Pclass']].apply(imputey, axis=1)
sns.heatmap(train.isnull(), yticklabels=False,cbar=False, cmap='viridis')

#Remove Cabin Column & Missing Embarked Row
train.drop('Cabin', axis=1,inplace=True)
train.dropna(inplace=True)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#Fix categorical columns - Male/Female need to be converted into a dummy variable 0 or 1, with column name as the value (Tidy Data)
sex = pd.get_dummies(train['Sex'], drop_first=True) #Drop female because it will lead to multicolinearity - one column is an exact predictor of the other column.
embark = pd.get_dummies(train['Embarked'], drop_first=True)

#Add these columns into the train dataframe
train=pd.concat([train, sex, embark], axis=1)

#Drop columns that we wont be using in our model
train.drop(['Embarked', 'Ticket', 'PassengerId', 'Name', 'Sex'], axis=1, inplace=True)

#### Build the Logistic Regression Model

# Define the dependent variable (y) and the independent variables (x)
x= train.drop('Survived', axis=1)
y= train['Survived']

# Create the Train/Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

#Build the Logistic Model

from sklearn.linear_model import LogisticRegression
logmod = LogisticRegression()

#Fit the xtrain data onto the Logistic Model
logmod.fit(x_train, y_train)

#### Make Predictions & Check Model
# Call some predictions on the x_test dataset
predictions = logmod.predict(x_test)

#Predictions is in np.array format but y_test is in Pd.DataFrame
y_test = np.array(y_test)

# Get the confusion matrix and the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))




"""
Logistic Regression Notes:

- It is a method for Classification.
- We try to predict which class an observation falls into.
- While linear regression predictions a continuous value, logistic regression predicts discrete classes.
- The model converts continues values into discrete categories.
- Convention for binary classification, 0 and 1.

Problems or Limitations of Linear Regression.
- For many datasets, LR wont make sense.
- For example, probably of getting into a university may go from negative to more than 1.
- Instead, we can transform the linear regression to a logistic regression curve.
- This can be done using the Sigmoid Function.
- The Sigmoid function takes in any value and outputs a value between 0 and 1.
- We make the cutoff point 0.5, hence anything above that belongs to the 1 class and anything below to the 0 class.

Model Evaluation & Confusion Matrix
- Predicted & True values are already known.
- True Positive = When we predict 1 and the answer is 1
- True Negative = When we predict 0 and the answer is 0
- False Positives (Type 1) = When we predict 1 and the answer is 0 (We predict a man is pregnant)
- True Negative (Type 2) = When we predict 0 and the answer is 1 (We predict a pregnant lady is not pregnant)

1. Accuracy = Overall how often is the model correct? (TP+TN)/TOTAL
2. Misclassification/Error Rate = Overall How often is the model wrong? (FP+FN)/TOTAL
3. Precision = When it predicts True, how often is it correct? (TP)/(TP+FP)
4. Recall/Sensitivity/True Positive Rate = When its actually True, how often does it predict True (TP/TP+FN)
"""