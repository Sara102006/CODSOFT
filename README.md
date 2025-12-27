import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading the dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Selection of useful features
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

# Handling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Convertion categorical data to numeric
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Splitting of features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Scalling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# model Evaluation
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Prediction of survival for a new passenger
# Passenger details:
# Pclass=3, Sex=female(1), Age=26, SibSp=0, Parch=0, Fare=7.25

new_passenger = np.array([[3, 1, 26, 0, 0, 7.25]])
new_passenger = scaler.transform(new_passenger)

prediction = model.predict(new_passenger)

print("\nNew Passenger Prediction:")
print("Survived" if prediction[0] == 1 else "Did not survive")
