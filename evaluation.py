import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from train import Train
from tensorflow import keras
from sklearn import metrics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    """
    이 함수는 
    https://michaelblogscode.wordpress.com/2017/12/20/visualizing-model-performance-statistics-with-tensorflow/
    로 부터 가져온 함수이며, 일부 수정하였습니다.
    """

    if normalize:
        cm = np.array(cm)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect=0.5)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = np.max(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=8,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    t = Train()
    model_path = './trained_model/sContrast/crnn_sContrast.h5'
    model = keras.models.load_model(model_path)
    print(model.summary())
    features = ['sContrast', 'c']
    t.read_data('./dataset/test_ds/', cols=features)

    y_test = np.array(t.data.pop('c'))
    print(y_test.shape)
    x_test = np.array([y for y in [x for x in t.data.pop(features[0]).values]])

    # shape = x_test.shape
    # x_test = x_test.reshape((shape[0], shape[1], 1))
    print(x_test.shape)

    y_pred = np.array([np.where(y==np.max(y))[0][0] for y in model.predict(x_test, batch_size=64)])

    print(y_pred.shape)

    cm = tf.math.confusion_matrix(y_test, y_pred, num_classes=30)
    print(cm)

    fig = plt.figure(figsize=(20, 20))
    plot_confusion_matrix(cm=cm, classes=t.class_names, normalize=True)
    plt.savefig(fname=model_path[:-3]+'_cm.png', bbox_inches='tight', dpi=300)

    f1_score = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(round(f1_score, 4))
