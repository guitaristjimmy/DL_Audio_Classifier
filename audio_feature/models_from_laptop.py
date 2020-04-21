import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


class Models:
    def __init__(self):
        pass

    def dnn(self, in_shape=(16000, 1)):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=256, kernel_size=320, strides=2, activation='relu', padding='causal',
                                input_shape=in_shape))
        model.add(layers.MaxPooling1D(pool_size=80))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(30, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def cnn1d(self, in_shape=(16000, 1)):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=256, kernel_size=320, activation='relu', input_shape=in_shape))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=64, kernel_size=160, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=128, kernel_size=80, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(30, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    # def cnn1d(self, in_shape=(16000, 1)):
    #     model = models.Sequential()
    #     model.add(layers.Conv1D(filters=32, kernel_size=320, activation='relu', input_shape=in_shape))
    #     model.add(layers.Conv1D(filters=32, kernel_size=320, activation='relu', input_shape=in_shape))
    #     model.add(layers.MaxPooling1D(pool_size=2))
    #     model.add(layers.Conv1D(filters=64, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=64, kernel_size=160, activation='relu'))
    #     model.add(layers.MaxPooling1D(pool_size=2))
    #     model.add(layers.Conv1D(filters=128, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=128, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=128, kernel_size=160, activation='relu'))
    #     model.add(layers.MaxPooling1D(pool_size=2))
    #     model.add(layers.Conv1D(filters=128, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=128, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=128, kernel_size=160, activation='relu'))
    #     model.add(layers.MaxPooling1D(pool_size=2))
    #     model.add(layers.Conv1D(filters=256, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=256, kernel_size=160, activation='relu'))
    #     model.add(layers.Conv1D(filters=256, kernel_size=160, activation='relu'))
    #     model.add(layers.MaxPooling1D(pool_size=8))
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(4096, activation='relu'))
    #     model.add(layers.Dropout(rate=0.25))
    #     model.add(layers.Dense(2048, activation='relu'))
    #     model.add(layers.Dropout(rate=0.25))
    #     model.add(layers.Dense(512, activation='relu'))
    #     model.add(layers.Dense(30, activation='softmax'))
    #
    #     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    #                   loss='categorical_crossentropy', metrics=['accuracy'])
    #     return model


if __name__ == '__main__':
    model = Models()
    dnn = model.dnn()
    print(dnn.summary())
    # cnn2d = model.cnn2d()
    # print(cnn2d.summary())
    # cnn1d = model.cnn1d()
    # print(cnn1d.summary())
