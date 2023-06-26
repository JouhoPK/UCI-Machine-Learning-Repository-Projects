import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing, model_selection

data = pd.read_csv("KNN/Iris/data.csv")

le = preprocessing.LabelEncoder()
s_length = le.fit_transform(list(data["sepal-length"]))
s_width = le.fit_transform(list(data["sepal-width"]))
p_length = le.fit_transform(list(data["petal-length"]))
p_width = le.fit_transform(list(data["petal-width"]))
cls = le.fit_transform(list(data["cls"]))

prediction = "cls"

X = list(zip(s_length, s_width, p_length, p_width))
y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
for x in range(len(x_test)):
    if predicted[x] == y_test[x]:
        print("Predicted: ", predicted[x], "Actual: ", y_test[x], "Correct!")

    else:
        print("Predicted: ", predicted[x], "Actual: ", y_test[x], "Wrong...")
