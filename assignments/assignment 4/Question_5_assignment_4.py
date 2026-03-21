from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#loading and splitting data
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



#decision tree from question 3
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=4,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)

print("Decision Tree Train Accuracy:", accuracy_score(y_train, dt_train_pred))
print("Decision Tree Test Accuracy:", accuracy_score(y_test, dt_test_pred))


#neural network from question 4
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPClassifier(
    hidden_layer_sizes=(16,),
    max_iter=500,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)

nn_train_pred = nn_model.predict(X_train_scaled)
nn_test_pred = nn_model.predict(X_test_scaled)

print("NN Train Accuracy:", accuracy_score(y_train, nn_train_pred))
print("NN Test Accuracy:", accuracy_score(y_test, nn_test_pred))

#question 5 confusion matrices
dt_cm = confusion_matrix(y_test, dt_test_pred)
nn_cm = confusion_matrix(y_test, nn_test_pred)

print("Decision Tree Confusion Matrix:\n", dt_cm)
print("Neural Network Confusion Matrix:\n", nn_cm)

#the neural network should be preferred for this task because it gets slightly
#   higher test accuracy and tends to make fewer classification errors like in the
#   confusion matrix

#Decision tree:
#advantage: easy to understand how decisions are made and which features are
#   important
#Limitation: it is more likely to overfit

#Neural Network:
#Advantage: predictive performance is strong and is able to capture complex patterns
#   in the data
#Limitation: it's more difficult to understand how the model is making its decisions
#   ("black box")