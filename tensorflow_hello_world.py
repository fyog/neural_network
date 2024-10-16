import tensorflow as tf
import numpy as np

# load dataset
mnist = tf.keras.datasets.mnist
(x_train, x_test), (y_train, y_test) = mnist.load_data() # MNIST dataset (range of 0 to 255, inclusive)
x_training, x_test = x_train / 255.0, x_test / 255.0 # adjust data range to 0 to 1, inclusive

# learning model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_training[:1]).numpy() # model returns a vector of logits (logarithm of the odds) for each class
predictions = tf.nn.softmax(predictions).numpy() # converts logits to probabilities for each class


# loss function
loss_f = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True) # loss is equal to the negative log probability of the true class
loss_f(y_train[:1], predictions).numpy()
print(loss_f)