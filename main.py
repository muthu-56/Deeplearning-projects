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

#Start of ANN
#Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, PReLU, ELU, Dropout

classifier = Sequential()

#Adding first input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(0.2))
#Adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform', activation = 'relu'))
classifier.add(Dropout(0.2))
#Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

#Compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model = classifier.fit(X_train, y_train, validation_split = 0.33, batch_size = 10, epochs = 100)


#model.history.keys()

#Predicting test data
y_pred = classifier.predict(X_test)
#y_pred =  (y_pred > 0.5)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)

