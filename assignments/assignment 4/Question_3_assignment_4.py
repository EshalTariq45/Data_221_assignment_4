from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#load dataset
data= load_breast_cancer()
X=data.data
y=data.target

#Train-test split
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Constrained Decision Tree
model= DecisionTreeClassifier(
    criterion= 'entropy',
    max_depth=4,   #constraint
    random_state=42
)

model.fit(X_train, y_train)

#Predictions
y_train_pred= model.predict(X_train)
y_test_pred= model.predict(X_test)

#Accuracy
train_acc=accuracy_score(y_train, y_train_pred)
test_acc= accuracy_score(y_test, y_test_pred)

print("Training Accuracy: ", train_acc)
print("Test accuracy: ", test_acc)

#feature importance
importances= model.feature_importances_
feature_names= data.feature_names

#Getting top 5 features
indices= np.argsort(importances)[::-1][:5]

print("\nTop 5 Important Features: ")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

#controlling model complexity helps prevent overfitting. A simpler tree cannot
#   memorize the training data as easily so its forced to learn more general-like
#   patterns. This would lower the training accuracy, but it would overall improve/
#   stabalize test accuracy

#In Q2, the tree is likely overfit. After hiding constraints the gap between training
#   and test accuracy becomes smaller, which indicates better generalization

#Feature importance shows how much each feature contributes to model's decisions.
#Features with higher importance are used more often or provide more information when
#   splitting
#this can make decision trees highly interpretable, since we can correctly identify which
#   of the variables have the most influence in predicting the outcome