from sklearn.datasets import load_breast_cancer
import numpy as np

#loading dataset
data=load_breast_cancer()

#feature matrix and target vector
X=data.data
y=data.target

#shape of X and y
print("shape of X: ", X.shape)
print("shape of y: ", y.shape)

#counting samples in each class
unique, counts= np.unique(y, return_counts=True)

for label, count in zip (unique, counts):
    print(f"class {label}: {count} samples")

#the dataset is a bit imbalanced because there are more benign samples
#   than malignant samples (357 vs 212)

#class balance is important because machine learning models may become biased
#   toward the majority class.

#In medical datasets, this is important because the minority class is often more
#   important to detect. Misclassifying these cases can have a negative impact on
#   the real-world