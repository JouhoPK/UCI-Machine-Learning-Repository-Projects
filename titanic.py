from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LogisticRegression
import pandas as pd

data = pd.read_csv("Titanic/titanic.csv")

le = preprocessing.LabelEncoder()
pclass = le.fit_transform(list(data["Pclass"]))
sex = le.fit_transform(list(data["Sex"]))
age = le.fit_transform(list(data["Age"]))
sibsp = le.fit_transform(list(data["SibSp"]))
parch = le.fit_transform(list(data["Parch"]))
cabin = le.fit_transform(list(data["Cabin"]))
embarked = le.fit_transform(list(data["Embarked"]))
cls = le.fit_transform(list(data["Survived"]))

PREDICT = "Survived"

X = list(zip(pclass, sex, age, sibsp, parch, cabin, embarked))
y = list(cls)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

model = LogisticRegression()
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predicted = model.predict(x_test)

print(acc * 100)
for x in range(len(x_test)):
    print("Predicted: ",  predicted[x], "Answer: ", y_test[x])
