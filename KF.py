'''
- Split the dataset into k folds:
The dataset is randomly shuffled and divided into k equal-sized subsets (folds). Common values for k are 5 or 10, but it can be any number.
- Training and testing process:
For each of the k folds:
One fold is used as the test set (to evaluate the model).
The remaining k-1 folds are used as the training set (to train the model).
- Repeat the process:
This process is repeated k times, each time with a different fold being used as the test set, while the remaining folds are used for training.
- Performance metric:
After all k iterations, the model's performance is averaged over all k test sets, providing a more robust estimate of its performance.
'''


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Initialize the model
model = RandomForestClassifier()

# Perform 5-fold cross-validation
#This function performs k-Fold Cross Validation for you. It takes a model (RandomForestClassifier), input data (X), labels (y), and the number of folds (cv=5).
scores = cross_val_score(model, X, y, cv=5)

# Print the cross-validation scores for each fold
print(f"Cross-validation scores for each fold: {scores}")

# Print the average score across all folds
print(f"Average score: {scores.mean()}")  #gives the average performance of the model across all the folds.



#and------> 


# Import necessary libraries
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier  # or any other model
from sklearn.datasets import load_iris  # Example dataset, you can replace it with your own data
from sklearn.model_selection import train_test_split
import numpy as np

# 1. Load dataset
data = load_iris()  # You can replace this with your own dataset
X = data.data
y = data.target

# 2. Initialize your model
model = RandomForestClassifier(random_state=42)

# 3. Define the number of splits (k)
k = 5  # You can choose any number, commonly 5 or 10

# 4. Create KFold object (splitting the data into k folds)
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 5. Apply k-Fold cross-validation (you can also use cross_val_score for simplicity)
cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')  # You can change scoring metric (e.g., 'f1', 'roc_auc', 'neg_log_loss').

# 6. Print the results
print(f"Cross-validation scores for each fold: {cv_scores}")
print(f"Average accuracy across all {k} folds: {np.mean(cv_scores)}")
print(f"Standard deviation of accuracy: {np.std(cv_scores)}")



#and-------->

#k-Fold Cross Validation for multiple machine learning models over different numbers of folds (5, 10, 20)

# Example Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42)
}

# Sample data (replace with your actual dataset)
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = load_iris()
x = data.data
y = data.target

# Scale data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Folds to use for cross-validation
folds = [5, 10, 20]

fig, axes = plt.subplots(len(models), len(folds), figsize=(15, 20))

for i, (model_name, model) in enumerate(models.items()):
    for j, fold in enumerate(folds):
        cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=42)
        accuracies = cross_val_score(model, x_scaled, y, cv=cv, scoring='accuracy')

        ax = axes[i, j]
        ax.plot(np.arange(1, fold+1), accuracies, marker='o', color='green')
        ax.set_title(f'{model_name} ({fold}-Fold)')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.grid(True)

plt.tight_layout()
fig.suptitle('Model Accuracy Across Different Folds', y=1.02)
plt.show()





#
