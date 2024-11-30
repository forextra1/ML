'''
ogistic Regression is a statistical method used for binary classification problems (i.e., classifying data into two categories).
It predicts the probability that an instance belongs to a particular class, and outputs a value between 0 and 1, which is mapped to the corresponding class.
'''

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load dataset (using the Iris dataset as an example)
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# To use Logistic Regression for binary classification, we will reduce it to two classes (e.g., Setosa and Versicolor)
X = X[y != 2]  # Remove class 2 (Virginica)
y = y[y != 2]  # Remove class 2 (Virginica)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model
logreg_model = LogisticRegression(random_state=42)

# Train the Logistic Regression model
logreg_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = logreg_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names) - 1)  # We removed class 2, so len is reduced
plt.xticks(tick_marks, iris.target_names[:2], rotation=45)
plt.yticks(tick_marks, iris.target_names[:2])

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Hyperparameter tuning using GridSearchCV (optional)
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga'], 'max_iter': [100, 200, 500]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and the corresponding accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Train a new model with the best parameters
best_logreg_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_logreg_model.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy (Best Model): {accuracy_best:.4f}')
print(f'Classification Report (Best Model):\n{classification_report(y_test, y_pred_best)}')





''''
Data Loading and Preprocessing:

The Iris dataset is used here, but for logistic regression, we reduce the dataset to a binary classification problem (by removing one of the classes). This allows logistic regression to be used as intended.
The data is split into training and test sets using train_test_split().
The features are standardized using StandardScaler() to ensure the model performs well.
Model Training:

We initialize the LogisticRegression model and train it on the scaled training data using fit().
Model Evaluation:

We make predictions using predict() and evaluate the model's performance using:
Accuracy: How many predictions were correct.
Classification Report: Shows precision, recall, and F1-score for each class.
Confusion Matrix: Helps visualize the number of correct and incorrect predictions.
Hyperparameter Tuning (Optional):

We use GridSearchCV to tune hyperparameters such as C (regularization strength), solver (optimization algorithm), and max_iter (number of iterations). This can help improve model performance.
Key Parameters in Logistic Regression:
C: Regularization strength. Smaller values of C mean more regularization (simpler model), while larger values of C result in less regularization (more complex model).
solver: The algorithm used for optimization. 'liblinear' is a good choice for small datasets, while 'saga' is better for large datasets.
max_iter: Maximum number of iterations for the solver. Some datasets may require more iterations for convergence.

'''

