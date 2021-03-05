# simple handwriting recognition model

# import libraries
import tensorflow as tf
from tensorflow import keras


# callback class
class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.9:
            print('\Reached 99% accuracy so cancelling training.')
            self.model.stop_training = True


# dataset
mnist = tf.keras.datasets.mnist

# split the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

# define model
model = tf.keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train_model
model.fit(x_train, y_train, verbose=1, epochs=10, callbacks=[callbacks])
