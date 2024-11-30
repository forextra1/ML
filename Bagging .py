'''
Bagging (Bootstrap Aggregating) is an ensemble learning technique used to improve the performance of machine learning algorithms, primarily decision trees. The primary goal of bagging is to reduce variance and prevent overfitting by creating multiple models from different subsets of the training data, then combining their predictions.

How Bagging Works:
Bootstrap Sampling: The data is sampled with replacement (i.e., some data points may appear multiple times, and others may not appear at all) to create several different subsets of the training dataset.
Model Training: A model is trained on each bootstrap sample (typically a weak learner, such as a decision tree).
Voting (for Classification) or Averaging (for Regression): After the models are trained, the predictions from all models are combined:
Classification: Each model's prediction is considered a "vote," and the class that receives the most votes across all models is chosen as the final prediction.
Regression: The predictions are averaged to obtain the final result.

'''
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset here)
data = load_iris()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a base classifier (Decision Tree)
base_model = DecisionTreeClassifier(random_state=42)

# Create the Bagging model using Decision Trees as base learners
bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=50, random_state=42)

# Train the Bagging model
bagging_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = bagging_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Bagging Model: {accuracy:.2f}")

# Plot the feature importances (for decision tree-based models like random forests)
plt.figure(figsize=(8, 6))
plt.bar(range(len(bagging_model.feature_importances_)), bagging_model.feature_importances_)
plt.title('Feature Importances from Bagging Model')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()


'''

Dataset: The Iris dataset is loaded using load_iris(), which contains four features and three target classes.
Train-Test Split: The dataset is split into a training set and a testing set using train_test_split().
Base Model: A decision tree classifier is used as the base learner (model trained on each bootstrap sample).
Bagging Model: The BaggingClassifier is used to combine multiple decision trees. The number of trees is set to 50, but this can be adjusted based on the problem.
Model Training: The bagging model is trained using the fit() method.
Model Evaluation: The model’s accuracy is calculated using accuracy_score() on the test data.
Feature Importances: Bagging models with decision trees can give insights into the relative importance of each feature, which is plotted here.

'''



