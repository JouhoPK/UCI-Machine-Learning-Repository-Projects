import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing, model_selection

data = pd.read_csv("KNN/Abalone/data.csv")

le = preprocessing.LabelEncoder()
sex = le.fit_transform(list(data["sex"]))
length = le.fit_transform(list(data["length"]))
diameter = le.fit_transform(list(data["diameter"]))
height = le.fit_transform(list(data["height"]))
whole_weight = le.fit_transform(list(data["whole_weight"]))
shucked_weight = le.fit_transform(list(data["shucked_weight"]))
viscera_weight = le.fit_transform(list(data["viscera_weight"]))
shell_weight = le.fit_transform(list(data["shell_weight"]))
rings = le.fit_transform(list(data["rings"]))

predict = "sex"

X = list(zip(length, diameter, height, whole_weight, shucked_weight, viscera_weight, shell_weight, rings))
y = list(sex)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.06)

model = KNeighborsClassifier(n_neighbors=13)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    if predicted[x] == 2:
        if y_test[x] == 2:
            print("Predicted: ", "Male", "Data: ", x_test[x], "Actual: ", "Male")

        elif y_test[x] == 0:
            print("Predicted: ", "Male", "Data: ", x_test[x], "Actual: ", "Female")

    if predicted[x] == 0:
        if y_test[x] == 2:
            print("Predicted: ", "Female", "Data: ", x_test[x], "Actual: ", "Male")

        if y_test[x] == 0:
            print("Predicted: ", "Female", "Data: ", x_test[x], "Actual: ", "Female")
