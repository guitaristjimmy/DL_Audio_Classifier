import pandas as pd
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models import Models
from tqdm import tqdm
# import time


class Train(Models):

    def __init__(self):
        self.data = pd.DataFrame()
        self.class_names = []

    def read_data(self, path, frac=1, cols=['c']):
        self.data = pd.DataFrame(columns=cols)
        d_cols = list(set(['n_audio', 'sCentroid', 'sContrast', 'stft', 'mel_spectrogram', 'mfcc', 'c'])-set(cols))
        paths = os.listdir(path=path)
        for p in tqdm(paths):
            self.class_names.append(p[:-4])
            df = pd.read_pickle(path+p)
            df = df.drop(columns=d_cols)
            if frac != 1:
                df = df.sample(frac=frac)
            self.data = self.data.append(df)
        self.data['c'] = pd.Categorical(self.data['c'])
        self.data['c'] = self.data.c.cat.codes
        print('read_data finish')


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MyCallBack, self).__init__()
        self.train_loss_logs, self.train_accuracy_logs, self.val_loss_logs, self.val_accuracy_logs = [], [], [], []
        self.epoch_num = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_num += 1

    def on_epoch_end(self, epoch, logs=None):
        # print(logs)
        self.train_loss_logs.append(logs['loss'])
        self.train_accuracy_logs.append(logs['categorical_accuracy']*100)
        self.val_loss_logs.append(logs['val_loss'])
        self.val_accuracy_logs.append(logs['val_categorical_accuracy']*100)

    # def on_train_batch_end(self, batch, logs=None):



if __name__ == '__main__':

    train = Train()
    my_callback = MyCallBack()
    features = ['mfcc', 'c']
    f = features[0]
    train.read_data('./dataset/valid_ds/', cols=features)
    y_valid = tf.keras.utils.to_categorical(np.array(train.data.pop('c')))
    print('y_valid ready')
    x_valid = np.array([y for y in [x for x in train.data.pop(f).values]])
    in_shape = x_valid.shape[1:]
    print('valid_ds ready ', x_valid.shape)

    model = train.crnn(in_shape=in_shape)
    print(model.summary())

    batch = 256

    for _ in range(0, 10):
        train.read_data('./dataset/train_ds/', frac=0.75, cols=features)

        y_train = tf.keras.utils.to_categorical(np.array(train.data.pop('c')))

        x_train = np.array([y for y in [x for x in train.data.pop(f).values]])

        model.fit(x=x_train, y=y_train, epochs=20, batch_size=batch, verbose=2, shuffle=True,
                  validation_data=(x_valid, y_valid), callbacks=[my_callback])

    print('model train finish')
    train.read_data('./dataset/test_ds/', cols=features)

    y_test = tf.keras.utils.to_categorical(np.array(train.data.pop('c')))

    x_test = np.array([y for y in [x for x in train.data.pop(f).values]])
    print(x_test.shape)
    print('test ds ready')
    model.evaluate(x=x_test, y=y_test, batch_size=batch)
    model.save(filepath='./trained_model/mfcc/crnn_mfcc.h5')

    # Plot -------------------------------------------------------------------------------------------------------------
    plt.subplot(1, 2, 1)
    plt.plot(my_callback.train_loss_logs, linestyle=':')
    plt.plot(my_callback.val_loss_logs)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(my_callback.train_accuracy_logs, linestyle=':')
    plt.plot(my_callback.val_accuracy_logs)
    plt.xlabel('epoch')
    plt.ylabel('%')
    plt.ylim((0, 100))
    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.show()
