# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset
# Replace this with your dataset loading code
# Example: df = pd.read_csv('your_dataset.csv')
df = pd.DataFrame({
    'Feature1': np.random.randn(100),
    'Feature2': np.random.randn(100),
    'Target': np.random.choice([0, 1], 100)  # Binary classification target
})

# 2. Data Preprocessing
X = df.drop('Target', axis=1)  # Features
y = df['Target']  # Target variable

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for algorithms like SVM, KNN, etc.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train the classification model
# Choose a classification model (e.g., Logistic Regression, Decision Tree, Random Forest)
# Logistic Regression (example)
model = LogisticRegression()

# Alternatively, you can try other models:
# model = DecisionTreeClassifier(random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = SVC(random_state=42)

# Fit the model
model.fit(X_train_scaled, y_train)

# 4. Model prediction
y_pred = model.predict(X_test_scaled)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# 6. Cross-validation (optional)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f'Cross-Validation Mean Accuracy: {cv_scores.mean():.4f}')

# 7. Hyperparameter tuning using GridSearchCV (optional)
param_grid = {
    'C': [0.1, 1, 10],  # For Logistic Regression or SVM
    'max_depth': [None, 10, 20, 30],  # For Decision Tree and Random Forest
    'n_estimators': [50, 100, 150],  # For Random Forest
    'kernel': ['linear', 'rbf']  # For SVM
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
# For other models, replace LogisticRegression() with your chosen model
grid_search.fit(X_train_scaled, y_train)

print(f'Best parameters: {grid_search.best_params_}')
best_model = grid_search.best_estimator_

# 8. Final predictions with the best model (if tuning was done)
final_predictions = best_model.predict(X_test_scaled)

# 9. Plotting confusion matrix (optional)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = ['Class 0', 'Class 1']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

# 10. Plotting ROC Curve (optional)
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



'''
Load the Dataset: You can replace the example dataset with your own dataset (e.g., loading a CSV file with pd.read_csv()).

Data Preprocessing:

Split the data into features (X) and the target variable (y).
Perform train-test split (80% train and 20% test).
Feature scaling using StandardScaler (important for algorithms like SVM, KNN, etc.).
Train the Classification Model:

You can choose from a variety of classifiers: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, SVC, etc.
Fit the model to the scaled training data.
Model Prediction:

Make predictions (y_pred) using the trained model on the test set.
Model Evaluation:

Accuracy: Evaluate the performance of the model using accuracy.
Confusion Matrix: Gives you insight into false positives and false negatives.
Classification Report: Provides precision, recall, F1-score, and support for each class.
Cross-validation (optional):

Perform cross-validation to evaluate the model’s performance across different subsets of the data.
Hyperparameter Tuning (optional):

Use GridSearchCV to search for the best hyperparameters (e.g., for LogisticRegression, SVM, DecisionTree, RandomForest).
Final Predictions:

Use the best model (after hyperparameter tuning, if applied) to make final predictions on the test set.
Plotting:

Confusion Matrix: Visualize the performance of the model.
ROC Curve: Plot the Receiver Operating Characteristic curve for binary classification (useful for evaluating classifiers).
'''
