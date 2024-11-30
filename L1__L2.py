'''
Regularization is a technique used in machine learning to prevent overfitting by adding a penalty to the loss function during training. 
L1 and L2 are the most common types of regularization.

What is Regularization?
Overfitting: When a model learns the noise in the training data instead of the underlying patterns, leading to poor generalization on unseen data.
Solution: Regularization discourages the model from becoming too complex by penalizing large weights, which helps in reducing overfitting.

'''

#L1 Lasso 
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate a sample dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L1 Regularization (Lasso Regression)
lasso = Lasso(alpha=0.1)  # alpha is the regularization strength
lasso.fit(X_train, y_train)

# Evaluate the model
print("Lasso Regression Coefficients:", lasso.coef_)
print("Lasso Regression R^2 on Test Data:", lasso.score(X_test, y_test))




#L2 Ridge 
from sklearn.linear_model import Ridge

# L2 Regularization (Ridge Regression)
ridge = Ridge(alpha=0.1)  # alpha is the regularization strength
ridge.fit(X_train, y_train)

# Evaluate the model
print("Ridge Regression Coefficients:", ridge.coef_)
print("Ridge Regression R^2 on Test Data:", ridge.score(X_test, y_test))





#Elastic Net:
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio controls balance between L1 and L2
elastic_net.fit(X_train, y_train)

# Evaluate the model
print("Elastic Net Coefficients:", elastic_net.coef_)
print("Elastic Net R^2 on Test Data:", elastic_net.score(X_test, y_test))





