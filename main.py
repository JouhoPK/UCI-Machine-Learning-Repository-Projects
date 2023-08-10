from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn import preprocessing, model_selection
import pandas as pd

data = pd.read_csv("data.csv")

le = preprocessing.LabelEncoder()
age = le.fit_transform(list(data["age"]))
year = le.fit_transform(list(data["year"]))
axillary = le.fit_transform(list(data["axillary"]))
cls = le.fit_transform(list(data["class"]))

PREDICT = "class"

X = list(zip(age, year, axillary))
y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

model = LogisticRegression()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predicted = model.predict(x_test)

print(acc * 100, '%')
for x in range(len(x_test)):
    print("Predicted: ", predicted[x], "Answer: ", y_test[x])