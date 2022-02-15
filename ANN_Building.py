import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# IMPORT DATASET

dataset = pd.read_csv("./data/Churn_Modelling.csv")

x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# print(x)
# print(y)

# ENCODING DATA

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# SPLITTING DATA

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FEATURE SCALING

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# BUILDING ANN

ann = tf.keras.models.Sequential()

# first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# relu - is used for fully connected layer
# sigmoid - is used for output layer because not only it gives the classificated output but also its probability
# softmax - is used when we have more than 2 categories


# TRAINING ANN

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# for binary classification we need to always use BINARY_CROSSENTROPY, when we have categorical model we use CATEGORICAL_CROSSENTROPY
# for metrics we can use multiple options accuracy is the basic one

ann.fit(x_train, y_train, batch_size=32, epochs=100)

# PREDICTING AND EVALUATING

# single test
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

# test set

y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
print(cm)

ac = accuracy_score(y_test, y_pred)
print(ac)