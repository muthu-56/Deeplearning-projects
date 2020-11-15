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

#Removing the encoded variable in the existed dataframe
df.drop(['Gender', 'Geography'], axis=1, inplace=True)

#Separating Dependent and independent feature
X=df['Exited']
Y = df.drop('Exited',axis=1)

print(Y)
