import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

#1. Load the dataset
(X_train, y_train), (X_test, y_test)= fashion_mnist.load_data()

#2. Preprocessing
# Normalize pixel values to be between 0 and 1
X_train= X_train/255.0
X_test= X_test/255.0

#Reshape to include the channel dimension (28,28,1) for grayscale
X_train= X_train.reshape((-1,28,28,1))
X_test= X_test.reshape((-1,28,28,1))

#3. Build the CNN model
model= models.Sequential([
    #Convolutional layer learns local patterns/features
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    #Maxpooling reduces spatial dimensions which provides translation invariance
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation= 'relu'),
    #Output layer for 10 classes
    layers.Dense(10, activation='softmax')
])

#4. Compile the model
model.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])

#5. Train the model
#Training for 15 epochs as requested
model.fit(X_train,y_train, epochs=15, validation_split=0.1)

#6. Report test accuracy
test_loss, test_acc= model.evaluate(X_test, y_test, verbose=2)


#question 7

#predictions
y_pred_probs=model.predict(X_test)
y_pred=np.argmax(y_pred_probs, axis=1)

#Confusion matrix
cm=confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

#showing misclassified indices
misclassified= np.where(y_pred!=y_test)[0]

#Class names (Fashion MNIST labels)
class_names= [
    'T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag', 'Ankle boot'
]

#showing 3 misclassified images
plt.figure(figsize=(10,4))

for i in range(3):
    idx=misclassified[i]

    plt.subplot(1,3,i+1)
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}")
    plt.axis('off')

plt.show()

#A common pattern in misclassification is thar visually similar items (an example
#   would be shirts vs pullovers or sneakers vs ankle boots) are confused.
#   this happens because these classes have shared similar shapes and textures

#One way to improve performance is to use a deeper CNN with more convolutional layers,
#   which can learn more complex features. Other options include data augmentation or
#   increasing the training time for data