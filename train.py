import datetime
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models import Models


class Train(Models):
    def __init__(self):
        pass

    def read_data(self, path, nums):
        paths = os.listdir(path=path)
        df = pd.DataFrame()
        for p in paths:
            class_id = p.split('/')[-1].replace('.pkl', '')
            if len(df.index) == 0:
                df = pd.read_pickle(path+p)
                df['c'] = class_id
                df = df.iloc[:nums]
            else:
                temp = pd.read_pickle(path+p)
                temp['c'] = class_id
                temp = temp.iloc[:nums]
                df = df.append(temp)
        return df


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        super(MyCallBack, self).__init__()
        self.train_loss_logs, self.train_accuracy_logs, self.val_loss_logs, self.val_accuracy_logs = [], [], [], []
        self.epoch_num = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_num += 1

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss_logs.append(logs['loss'])
        self.train_accuracy_logs.append(logs['accuracy']*100)
        self.val_loss_logs.append(logs['val_loss'])
        self.val_accuracy_logs.append(logs['val_accuracy']*100)

    def on_train_batch_end(self, batch, logs=None):
        print('\t epoch :: ', self.epoch_num)

    def on_test_end(self, logs=None):
        print('test logs :: ', logs)

if __name__ == '__main__':

    train = Train()
    data = train.read_data('./dataset/train_prepared/', 1024)
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes
    # label = data.pop('c')
    x_train = np.array([np.array(z).T for z in [[y] for y in [x for x in data['n_audio'].values]]])
    in_shape = x_train.shape[1:]
    y_train = tf.keras.utils.to_categorical(np.array(data['c']))
    print('ds ready')

    data = train.read_data('./dataset/valid_prepared/', 256)
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes
    # label = data.pop('c')
    x_valid = np.array([np.array(z).T for z in [[y] for y in [x for x in data['n_audio'].values]]])
    y_valid = tf.keras.utils.to_categorical(np.array(data['c']))
    data = False
    print('val ds ready')
    model = train.cnn1d(in_shape=in_shape)
    callback_logs = MyCallBack()
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=64, validation_data=(x_valid, y_valid),
              callbacks=[callback_logs], shuffle=True)

    print('model train finish')
    data = train.read_data('./dataset/test_prepared/', 1000)
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes

    x_test = np.array([np.array(z).T for z in [[y] for y in [x for x in data['n_audio'].values]]])
    y_test = tf.keras.utils.to_categorical(np.array(data['c']))
    print('test ds ready')
    model.evaluate(x=x_test, y=y_test, batch_size=64)
    model.save(filepath='./trained_model/200315_00.h5')
    plt.plot(callback_logs.train_loss_logs, linestyle=':')
    plt.plot(callback_logs.train_accuracy_logs)
    plt.plot(callback_logs.val_loss_logs, linestyle=':')
    plt.plot(callback_logs.val_accuracy_logs)
    plt.xlabel('epoch')
    plt.ylabel('%')
    plt.legend(['train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
    plt.show()
