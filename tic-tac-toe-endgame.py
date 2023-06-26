import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing, model_selection

data = pd.read_csv("KNN/TicTacToe/data.csv")

le = preprocessing.LabelEncoder()
top_left_square = le.fit_transform(list(data["top-left-square"]))
top_middle_square = le.fit_transform(list(data["top-middle-square"]))
top_right_square = le.fit_transform(list(data["top-right-square"]))
middle_left_square = le.fit_transform(list(data["middle-left-square"]))
middle_middle_square = le.fit_transform(list(data["middle-left-square"]))
middle_right_square = le.fit_transform(list(data["middle-left-square"]))
bottom_left_square = le.fit_transform(list(data["bottom-left-square"]))
bottom_middle_square = le.fit_transform(list(data["bottom-middle-square"]))
bottom_right_square = le.fit_transform(list(data["bottom-right-square"]))
cls = le.fit_transform(list(data["class"]))

prediction = "cls"

X = list(zip(top_left_square, top_middle_square, top_right_square, middle_left_square, middle_middle_square, middle_right_square, bottom_left_square, bottom_middle_square, bottom_right_square))
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
