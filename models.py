import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


class Models:
    def __init__(self):
        pass

    def dnn(self, in_shape=(16000, 1)):
        model = models.Sequential()
        model.add(layers.Dense(4096, activation='relu', input_shape=in_shape))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1012, activation='relu'))
        model.add(layers.Dense(30, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def cnn2d(self, in_shape=(256, 256, 3)):    #VGG16
        model = models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dense(1000, activation='relu'))
        model.add(layers.Dense(30, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # def cnn1d(self, in_shape=(16000, 1)):
    #     #     model = models.Sequential()
    #     #     model.add(layers.Conv1D(filters=512, kernel_size=160, strides=16, padding='causal', activation='relu',
    #     #                             input_shape=in_shape))
    #     #     model.add(layers.Conv1D(filters=512, kernel_size=10, strides=10, padding='causal', activation='relu',
    #     #                             input_shape=in_shape))
    #     #     model.add(layers.AveragePooling1D(pool_size=2))
    #     #     model.add(layers.Flatten())
    #     #     model.add(layers.Dense(2048, activation='relu'))
    #     #     model.add(layers.Dropout(0.5))
    #     #     model.add(layers.Dense(2048, activation='relu'))
    #     #     model.add(layers.Dropout(0.5))
    #     #     model.add(layers.Dense(256, activation='relu'))
    #     #     model.add(layers.Dropout(0.5))
    #     #     model.add(layers.Dense(30, activation='softmax'))
    #     #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     #                   loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])
    #     #     return model

    def cnn1d(self, in_shape=(16000, 1)):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=32, kernel_size=256, activation='relu', input_shape=in_shape))
        model.add(layers.Conv1D(filters=32, kernel_size=128, activation='relu'))
        model.add(layers.Conv1D(filters=32, kernel_size=128, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=64, kernel_size=128, activation='relu'))
        model.add(layers.Conv1D(filters=64, kernel_size=128, activation='relu'))
        model.add(layers.Conv1D(filters=64, kernel_size=128, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(30, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model


if __name__ == '__main__':
    model = Models()
    # dnn = model.dnn()
    # print(dnn.summary())
    # cnn2d = model.cnn2d()
    # print(cnn2d.summary())
    cnn1d = model.cnn1d()
    print(cnn1d.summary())
