'''
Boosting is an ensemble learning technique that combines multiple weak learners to create a strong learner. 
The key idea behind boosting is to build models sequentially, where each new model focuses on the errors made by the previous model, 
thereby improving the model's performance. Boosting is particularly effective for classification and regression problems and 
is known for its ability to significantly increase model accuracy.

Sequential Learning: Boosting builds multiple models sequentially. The first model is trained on the data, and the second model is trained to correct the errors of the first model, 
the third model is trained to correct the errors of the combined first and second models, and so on.
Error Correction: Each subsequent model in the sequence gives more importance to the data points that were misclassified by the previous models.
Weighting: In boosting, each model's contribution to the final prediction is weighted based on its performance (e.g., more weight is given to better-performing models).
Final Prediction: The final prediction is made by combining the predictions of all the models, with each model's contribution weighted according to its performance.

'''


# Import necessary libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a base model (Decision Tree)
base_model = DecisionTreeClassifier(max_depth=1, random_state=42)  # A weak learner (stump)

# Initialize AdaBoost with the base model
adaboost_model = AdaBoostClassifier(base_estimator=base_model, n_estimators=50, random_state=42)

# Train the AdaBoost model
adaboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = adaboost_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of AdaBoost Model: {accuracy:.2f}")

'''
Dataset: The Iris dataset is used for classification, where the goal is to classify flowers into three species.
Train-Test Split: The dataset is split into training and test sets using train_test_split().
Base Model: A Decision Tree with a maximum depth of 1 is used as the base learner. This is a "weak" model, which will be iteratively improved by AdaBoost.
AdaBoost: The AdaBoostClassifier is used with the base model (base_estimator). We use 50 estimators (iterations).
Training and Prediction: The AdaBoost model is trained using fit() on the training data, and predictions are made using predict().
Evaluation: The model's accuracy is calculated using accuracy_score().

'''






from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gradient Boosting Classifier
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gb_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Gradient Boosting Model: {accuracy:.2f}")


'''
n_estimators: Number of weak learners (models) to train.
learning_rate: Controls the contribution of each model to the final prediction. A smaller learning rate means the model is trained more conservatively.
max_depth (for Decision Trees in AdaBoost or Gradient Boosting): Controls the depth of the individual weak learners (trees).
subsample: Fraction of samples used to fit each base learner (only in Gradient Boosting and similar models).

'''
