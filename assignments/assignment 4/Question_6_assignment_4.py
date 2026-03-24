import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist

#1. Load the dataset
(X_train, y_train), (X_test, y_test)= fashion_mnist.load_data()

#2. Preprocessing
# Normalize pixel values to be between 0 and 1
X_train= X_train.astype("float32")/255.0
X_test= X_test.astype("float32")/255.0

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
history= model.fit(X_train,y_train, epochs=15, validation_data=(X_test, y_test))

#6. Report test accuracy
test_loss, test_acc= model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc:.4f}")

#explanation:

#   1. Why CNNs are preferred over fully connected (Dense) networks for image data:
# parameter efficiency- CNNs use wight sharing. A filter detects an edge in the top
#   corner is the same filer used for the bottom-right. Denser networks would require
#   unique weights for every single pixel-to-neuron connection.
#Spatial Hierarchy- CNNs preserve the 2D structure of images. Denser networks would
#   require 'flattening' the image into a 1D vector, which would destroy the spatial
#   relationship between the neighbouring pixels

#   2. What the convolution layer is learning in this task:
#In early layers, the Conv2D filters are learning the lower level features like
#   horizontal/vertical edges, simple textures and corners. as the data moves deeper,
#   these features are combined to recognize more complex shapes such as sleeves,
#   collars, etc
