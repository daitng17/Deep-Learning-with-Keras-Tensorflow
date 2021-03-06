# simple cnn model

# import libraries
import tensorflow as tf
from tensorflow import keras


# define callback class
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.998:
            print("\nReached 99.8% so cancelling training!")
            self.model.stop_training = True


callback = myCallback()

# get the data
mnist = keras.datasets.mnist

# split the dataset to training set and test set
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshape the images
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255.0

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# define model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(train_images, train_labels, verbose=1, epochs=20, callbacks=[callback])