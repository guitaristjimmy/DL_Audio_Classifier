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
        print(logs)
        self.train_loss_logs.append(logs['loss'])
        self.train_accuracy_logs.append(logs['categorical_accuracy']*100)
        self.val_loss_logs.append(logs['val_loss'])
        self.val_accuracy_logs.append(logs['val_categorical_accuracy']*100)

    def on_train_batch_end(self, batch, logs=None):
        print('\t epoch :: ', self.epoch_num)


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

    data = train.read_data('./dataset/valid_prepared/', 512)
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes
    # label = data.pop('c')
    x_valid = np.array([np.array(z).T for z in [[y] for y in [x for x in data['n_audio'].values]]])
    y_valid = tf.keras.utils.to_categorical(np.array(data['c']))
    data = False
    print('val ds ready')
    model = train.cnn1d(in_shape=in_shape)
    callback_logs = MyCallBack()

    model.fit(x=x_train, y=y_train, epochs=30, batch_size=64, verbose=1, shuffle=True,
              validation_data=(x_valid, y_valid), callbacks=[callback_logs])

    print('model train finish')
    data = train.read_data('./dataset/test_prepared/', 1024)
    data['c'] = pd.Categorical(data['c'])
    data['c'] = data.c.cat.codes

    x_test = np.array([np.array(z).T for z in [[y] for y in [x for x in data['n_audio'].values]]])
    y_test = tf.keras.utils.to_categorical(np.array(data['c']))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)
    print('test ds ready')
    model.evaluate(test_ds)
    model.save(filepath='./trained_model/cnn1d.h5')
    plt.subplot(1, 2, 1)
    plt.plot(callback_logs.train_loss_logs, linestyle=':')
    plt.plot(callback_logs.val_loss_logs)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.subplot(1, 2, 2)
    plt.plot(callback_logs.train_accuracy_logs, linestyle=':')
    plt.plot(callback_logs.val_accuracy_logs)
    plt.xlabel('epoch')
    plt.ylabel('%')



















    plt.legend(['train_accuracy', 'val_accuracy'])
    plt.show()
