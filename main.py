#Importing libraries
import numpy as np
import pandas as pd

#Reading data
df = pd.read_csv('D:/Mari/Data science material/Additional - Projects/DL/Churn model/Churn_Modelling.csv')
df.head()

#Data summary
df.shape
df.info()

#Rownumber, customerID and Surname are having more unique values, so we can ignore this
df.drop(['RowNumber','CustomerId','Surname'], axis=1, inplace=True)
df.dtypes

#Geography and Gender variable encoding
enc=pd.get_dummies(df[['Geography','Gender']], drop_first=True)

#Removing the already encoded variable in the existing dataframe
df.drop(['Gender', 'Geography'], axis=1, inplace=True)

#Concat encoded variable into existing dataframe
df = pd.concat([df,enc], axis=1)

#Separating Dependent and independent feature
Y = df['Exited']
X = df.drop('Exited',axis=1)

#Splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

