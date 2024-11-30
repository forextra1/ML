'''
Stacking (also called stacked generalization) is an ensemble learning technique that combines multiple base models (also known as level-0 models) by training
a meta-model (level-1 model) to make final predictions. The key idea behind stacking is to use a meta-model to learn how to 
best combine the predictions of several base models, improving the overall performance.

Base Models: You train multiple different base models (e.g., decision trees, logistic regression, etc.) on the training dataset.
Predictions: Each base model makes predictions on the training dataset.
Meta-Model: A second-level model (meta-model) is trained on the predictions made by the base models. The meta-model learns how to combine the outputs of the base models to improve performance.
Final Prediction: Once the meta-model is trained, it can make predictions on unseen data using the predictions from the base models.
The key benefit of stacking is that different models may have complementary strengths, and the meta-model can learn to exploit those differences to make better predictions.

Types of Stacking:
Stacking for Classification: The meta-model predicts a class label based on the base model predictions.
Stacking for Regression: The meta-model predicts a continuous value (regression) based on the base model predictions.
'''

# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a sample dataset (Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the base models
base_learners = [
    ('decision_tree', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(kernel='linear', random_state=42)),
    ('logistic_regression', LogisticRegression(max_iter=200, random_state=42))
]

# Define the meta-model (model to combine the base model predictions)
meta_model = LogisticRegression(random_state=42)

# Create the Stacking model (ensemble of base models)
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

# Train the stacking model
stacking_model.fit(X_train, y_train)

# Make predictions using the stacking model
y_pred = stacking_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Stacking Model: {accuracy:.2f}")



'''
Dataset: The Iris dataset is used for classification, where the goal is to predict the species of flowers based on their features.
Train-Test Split: The dataset is split into training and test sets using train_test_split().
Base Models: Three base models are defined:
DecisionTreeClassifier
SVC (Support Vector Classifier) with a linear kernel
LogisticRegression
Meta-Model: The meta-model (final estimator) is a Logistic Regression model. This model will learn how to combine the predictions from the base models.
Stacking Model: The StackingClassifier from sklearn.ensemble is used to combine the base models. The final_estimator is the meta-model that combines predictions.
Model Training: The stacking model is trained on the training data using fit().
Prediction and Evaluation: The model makes predictions on the test data, and the accuracy of the predictions is calculated using accuracy_score().
Key Hyperparameters for StackingClassifier:
estimators: List of tuples, where each tuple contains a name and a base model. These are the models that will be used to make predictions before the meta-model is applied.
final_estimator: The meta-model that takes the predictions from the base models and makes the final prediction.
stack_method: Determines how the base models' predictions are passed to the meta-model. Options include:
auto: Uses the default behavior.
predict_proba: Passes probability estimates to the meta-model (for classification tasks).
predict: Passes class predictions to the meta-model.

'''

