import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"-----")


features=df.columns[:-1]

#selected_features =np.random.choice(features, 3, replace=False)
selected_features = ['SmallDependenceLowGrayLevelEmphasis', 'MajorAxisLength', 'ZoneVariance']

plt.figure(figsize=(10, 4))

for i, feature in enumerate(selected_features):
    plt.subplot(1, 3, i + 1)  
    sns.histplot(df[feature], kde=True, bins=15)  
    plt.title(f'{feature}')
    plt.xlabel(feature )
    
plt.tight_layout()
plt.show()




corr_matrix = df.corr()

plt.figure(figsize=(8, 6))  
sns.heatmap(corr_matrix, cmap='coolwarm')


 
#>>strong correlations
threshold = 0.9 
strong_corr = set()  

for column in corr_matrix.columns:
    for row in corr_matrix.index:
        if abs(corr_matrix.loc[row, column]) > threshold and column != row:
            pair = tuple(sorted([column, row]))
            strong_corr.add((pair, corr_matrix.loc[row, column]))

strong_corr = list(strong_corr)  
strong_corr.sort(key=lambda x: abs(x[1]), reverse=True)  

for pair, correlation in strong_corr[:10]:
    print(f"-> {pair[0]} and {pair[1]} have a correlation of {correlation:.6f}")





from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR

x = df.iloc[:, :-1]
y = df.iloc[:, -1]   

x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sample_sizes = [0.20, 0.50, 0.80]  

models = {
    "SVM": SVC(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Bagging Classifier": BaggingClassifier(random_state=42),
    "AdaBoost Classifier": AdaBoostClassifier(algorithm='SAMME', random_state=42),  # Updated line
    "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42),
    "XGBoost Classifier": xgb.XGBClassifier(random_state=42),
    "Stacking Classifier": StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=10, random_state=42)), 
                                                          ('lr', LR())], final_estimator=LogisticRegression())
}


scaler = StandardScaler()
results = []

for sample in sample_sizes:
    x_subset, _, y_subset, _ = train_test_split(x_train_full, y_train_full, train_size=sample, random_state=42)
    
    X_scaled = scaler.fit_transform(x_subset)
    X_test_scaled = scaler.transform(x_test)
    
    for model_name, model in models.items():
        model.fit(X_scaled, y_subset)
        
        y_pred = model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results.append({
            "Model": model_name,
            "Sample Size": f"{int(sample * 100)}%",
            "Accuracy": f"{accuracy:.3f}",
            "Precision": f"{report['weighted avg']['precision']:.3f}",
            "Recall": f"{report['weighted avg']['recall']:.3f}",
            "F1-Score": f"{report['weighted avg']['f1-score']:.3f}"
        })

results_df = pd.DataFrame(results)
results_df











from sklearn.decomposition import PCA


scaler = StandardScaler()
x_scaled = scaler.fit_transform(df.iloc[:, :-1])  

pca = PCA().fit(x_scaled)

explained_variance_ratio = pca.explained_variance_ratio_

cumulative_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('number of components')
plt.ylabel('cumulative explained var')
plt.title('cumulative explained var vs number of components')
plt.axhline(y=0.90, color='r')  # 90%  var
plt.axhline(y=0.95, color='g')  #  95%  var
plt.show()

n_components_90 = (cumulative_variance >= 0.90).argmax() + 1
n_components_95 = (cumulative_variance >= 0.95).argmax() + 1

print(f"optimal number of components for 90% var: {n_components_90}")
print(f"optimal number of components for 95% var: {n_components_95}")














model = LogisticRegression(solver='saga', max_iter=1000, random_state=42, class_weight='balanced')

training_sizes = np.linspace(0.1, 1.0, 10)

train_accuracies = []
test_accuracies = []

for size in training_sizes:
    x_train_subset = x_train_full[:int(size * len(x_train_full))]
    y_train_subset = y_train_full[:int(size * len(y_train_full))]
    
    X_train_scaled = scaler.fit_transform(x_train_subset)
    X_test_scaled = scaler.transform(x_test)

    model.fit(X_train_scaled, y_train_subset)
    
    train_accuracy = accuracy_score(y_train_subset, model.predict(X_train_scaled))
    test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

plt.figure(figsize=(6, 4))
plt.plot(training_sizes * len(x_train_full), train_accuracies, label='Training Accuracy', marker='o')
plt.plot(training_sizes * len(x_train_full), test_accuracies, label='Test Accuracy', marker='o')
plt.xlabel('training Size')
plt.ylabel('accuracy')
plt.title('learning curve for Logistic Regression')
plt.legend()
plt.grid(True)
plt.show()















from sklearn.model_selection import StratifiedKFold, cross_val_score

x_scaled = scaler.fit_transform(x)

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
















