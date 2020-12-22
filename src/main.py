import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import os

fileDir = os.path.dirname(__file__)
filename = os.path.join(fileDir, '../resources/iris.csv')
filename = os.path.abspath(os.path.realpath(filename))
print(filename)
df = pd.read_csv(filename)

print(df["species"].unique())

df["species"] = df["species"].map(
	{
		"Iris-setosa": 0,
		"Iris-versicolor": 1,
		"Iris-virginica": 2
	}
) 
print(df.head)


X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy()
y = df[["species"]].to_numpy()

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'kernel': ['rbf', 'linear'], 'C':[0.1,1,10], 'gamma':[1,0.1,0.01]}
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=3)
grid.fit(X_train, y_train)

pred = grid.predict(X_test)
print(classification_report(y_test,pred))