import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
print(tf.__version__)

dataset = pd.read_csv('train.csv')
# print(dataset.head())

X = dataset.drop(labels=['Sales'], axis = 1)
y = dataset['Sales']

# print(X.head())
label1 = LabelEncoder()
X['Item_ID'] = label1.fit_transform(X['Item_ID'])

label = LabelEncoder()
X['Item_Type'] = label.fit_transform(X['Item_Type'])

label2 = LabelEncoder()
X['Outlet_Size'] = label2.fit_transform(X['Outlet_Size'])

label3 = LabelEncoder()
X['Outlet_ID'] = label3.fit_transform(X['Outlet_ID'])

label4 = LabelEncoder()
X['Outlet_Location_Type'] = label4.fit_transform(X['Outlet_Location_Type'])
print(X.head())
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()
model.add(Dense(X.shape[1], activation='relu', input_dim = X.shape[1]))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train.to_numpy(), batch_size = 10, epochs = 10, verbose = 1)

y_pred = (model.predict(X_test) > 0.5).astype("int32")

# y_pred = model.predict(X_test)
# print(y_pred)
# print(y_test)
# model.evaluate(X_test, y_test.to_numpy())
# from sklearn.metrics import confusion_matrix, accuracy_score
# print(confusion_matrix(y_test, y_pred))
# print(accuracy_score(y_test, y_pred))