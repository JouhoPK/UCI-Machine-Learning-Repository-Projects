import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, model_selection

data = pd.read_csv("KNN/Glass/data.csv")

id_number = list(data["Id_number"])
ri = list(data["RI"])
na = list(data["Na"])
mg = list(data["Mg"])
al = list(data["Al"])
si = list(data["Si"])
k = list(data["K"])
ca = list(data["Ca"])
ba = list(data["Ba"])
fe = list(data["Fe"])
cls = list(data["Class"])

predict = "Class"
X = list(zip(id_number))
y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
