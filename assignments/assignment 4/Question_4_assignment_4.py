from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#loading datasets
data= load_breast_cancer()
X=data.data
y=data.target

#Train-test split
X_train, X_test, y_train, y_test= train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Standardize features
scaler= StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Neural Network
model= MLPClassifier(
    hidden_layer_sizes=(16,), #one hidden layer with 16 neurons
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

model.fit(X_train_scaled, y_train)

#predictions
y_train_pred= model.predict(X_train_scaled)
y_test_pred= model.predict(X_test_scaled)

#Accuracy
train_acc= accuracy_score(y_train, y_train_pred)
test_acc= accuracy_score(y_test, y_test_pred)

print("Training accuracy: ", train_acc)
print("Test accuracy: ", test_acc)

#Feature scaling is necessary for neural networks because they rely on gradient-
#   based optimization. If features are on a very different scale, the model may
#   start to converge slowly or even get stuck.

#an epoch represents one complete pass through the entire training dataset during
#   training. In each epoch the model updates weights based on all training examples
#   multiple epocks are needed in order for the model to learn patterns in data