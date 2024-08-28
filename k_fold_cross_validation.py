import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Number of folds
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=200)
accuracies = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)

print(f"Accuracies for each fold: {accuracies}")
print(f"Mean accuracy: {np.mean(accuracies)}")

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(range(1, k+1), accuracies, color='skyblue')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Accuracy per Fold in K-Fold Cross-Validation')
plt.xticks(range(1, k+1))
plt.ylim(0, 1)
plt.grid(axis='y')
plt.show()
