import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

data = pd.read_csv("LinearRegression/StudentPerformance/data.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "Medu", "Fedu", "traveltime"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
print(X)
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])