from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#loading dataset
data= load_breast_cancer()
X= data.data
y=data.target

#80/20 split
X_train, X_test, y_train, y_test= train_test_split(
    X,y,test_size=0.2, stratify=y, random_state=42
)

#train decision tree with entropy
model= DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train,y_train)

#predictions
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

#accuracy
train_acc=accuracy_score(y_train, y_train_pred)
test_acc= accuracy_score(y_test, y_test_pred)

print("Training accuracy: ", train_acc)
print("Test accuracy: ", test_acc)

#entropy measures the amount of uncertainty or randomness in the data
#in decision trees it's used to decide how to split the data at each node
#a split with lower entropy means the data becomes more 'pure'
#the model chooses splits that maximize information gain (reduce entropy at
#   the most)>

#training accuracy is very high while test accuracy is lower
#this suggests that the model may be overfitting the training data
#overfitting happens when the model memorizes the training data instead of
#   learning.
#general patterns leading to slightly worse performance on unseen data

#since the test accuracy is still high, the model still generalizes well,
#   but it could benefit from limiting the tree depth