import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample Credit Scoring Dataset
data = {
    'income': [25000, 40000, 50000, 20000, 60000, 35000, 80000, 30000],
    'loan_amount': [10000, 15000, 20000, 12000, 25000, 10000, 30000, 18000],
    'credit_score': [600, 700, 750, 550, 800, 650, 780, 580],
    'approved': [1, 1, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[['income', 'loan_amount', 'credit_score']]
y = df['approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Credit Scoring Model")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))
