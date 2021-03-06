import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


class Models:
    def __init__(self):
        pass

    def dnn(self, in_shape=(32000, 3)):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=in_shape))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(30, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def cnn2d(self, in_shape=(32, 16, 1)):
        model = models.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=in_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        # model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(30, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
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

    def cnn1d(self, in_shape=(512, 1)):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='causal', input_shape=in_shape))
        model.add(layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='causal'))
        model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(30, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def rnn(self, in_shape=(512, 1)):
        model = models.Sequential()
        model.add(layers.LSTM(256, return_sequences=True, input_shape=in_shape,
                              activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
                              use_bias=True))
        model.add(layers.Dropout(0.5))
        model.add(layers.TimeDistributed(layers.Dense(128, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dense(32, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dense(16, activation='relu')))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(30, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def crnn(self, in_shape=(512, 1)):
        model = models.Sequential()
        model.add(layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='causal', input_shape=in_shape))
        model.add(layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='causal'))
        model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='causal'))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(256, return_sequences=True,
                              activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
                              use_bias=True))
        model.add(layers.Dropout(0.5))
        model.add(layers.TimeDistributed(layers.Dense(128, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dense(64, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dense(32, activation='relu')))
        model.add(layers.TimeDistributed(layers.Dense(16, activation='relu')))
        model.add(layers.Flatten())
        model.add(layers.Dense(30, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model



if __name__ == '__main__':
    model = Models()
    shape = (16000, 1)
    # dnn = model.dnn(in_shape=shape)
    # print(dnn.summary())
    # cnn2d = model.cnn2d(in_shape=shape)
    # print(cnn2d.summary())
    # cnn1d = model.cnn1d(in_shape=shape)
    # print(cnn1d.summary())
    # rnn = model.rnn(in_shape=shape)
    # print(rnn.summary())
    crnn = model.crnn(in_shape=shape)
    print(crnn.summary())
