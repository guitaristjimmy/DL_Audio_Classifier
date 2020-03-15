from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

train, test = tf.keras.datasets.fashion_mnist.load_data()
print(train)

images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))