'''
Support Vector Machine (SVM) is a supervised machine learning algorithm commonly used for classification and regression tasks.
It is particularly effective for classification tasks, especially when the data is not linearly separable.
'''


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM classifier
svm_model = SVC(kernel='linear')  # Linear kernel (you can change the kernel)
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')



#and------> 


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load dataset (using the Iris dataset as an example)
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model with a linear kernel (you can change the kernel if needed)
svm_model = SVC(kernel='linear', random_state=42)

# Train the SVM model
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test data
y_pred = svm_model.predict(X_test_scaled)

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
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Hyperparameter tuning using GridSearchCV (optional)
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Display the best parameters and the corresponding accuracy
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

# Train a new model with the best parameters
best_svm_model = grid_search.best_estimator_

# Make predictions with the best model
y_pred_best = best_svm_model.predict(X_test_scaled)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Accuracy (Best Model): {accuracy_best:.4f}')
print(f'Classification Report (Best Model):\n{classification_report(y_test, y_pred_best)}')


''''
Data Loading and Preprocessing:

We load the dataset using datasets.load_iris() from scikit-learn. You can replace this with your own dataset.
The dataset is split into training and testing sets using train_test_split().
We scale the features using StandardScaler(), which is important for SVM to perform well.
Model Training:

We create an instance of SVC (Support Vector Classification) with a linear kernel.
We train the model on the scaled training data using fit().
Model Evaluation:

We make predictions on the test set using the predict() method.
The model's performance is evaluated using accuracy_score(), classification_report(), and a confusion matrix.
Hyperparameter Tuning (Optional):

We perform hyperparameter tuning using GridSearchCV to find the best combination of parameters (C, gamma, and kernel).
After finding the best model, we train and evaluate it again.
Visualization:

The confusion matrix is visualized using matplotlib to better understand where the model is making classification errors.
Explanation of Key Parameters:
kernel='linear': This specifies that we are using a linear kernel. You can change it to 'rbf' for a radial basis function kernel or other kernels depending on the data.
C: Regularization parameter that controls the trade-off between achieving a low training error and a low testing error. Higher values of C make the decision boundary more specific to the training data, while lower values make it smoother.
gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. 'scale' (default) uses 1 / (n_features * X.var()), and 'auto' uses 1 / n_features.

'''




