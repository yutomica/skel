
## NN
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras import optimizers

import numpy as np
from sklearn.grid_search import GridSearchCV
from keras.utils import np_utils
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier

# auc
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


# CV用モデル
def model(activation="relu",optimizer="adam",n_in=len(X_train.columns),n_hiddens=[50,50],n_out=1,p_keep=0.01):
    model = Sequential()
    for i,input_dim in enumerate(([n_in]+n_hiddens)[:-1]):
        model.add(Dense(n_hiddens[i],input_dim=input_dim))
        model.add(Activation(activation))
        model.add(Dropout(p_keep))
    model.add(Dense(n_out))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy',auc_roc])
    return model
early_stopping = EarlyStopping(monitor='val_loss',patience=10,verbose=1)

# パラメータ定義
n_hiddens = [[60,60],[70,70],[80,80],[90,90],[100,100]]
#n_hiddens = [[5,5,5],[10,10,10],[5,5,5,5],[10,10,10,10]]

activation = ["relu", "sigmoid"]
optimizer = ["adam", "adagrad"]

nb_epoch = [10, 25]
batch_size = [5, 10]

# 層数の決定
clf = KerasClassifier(build_fn=model, verbose=0)
param_grid = [
    #{'n_hiddens':n_hiddens,'p_keep':[0.01]},
    #{'n_hiddens':n_hiddens,'p_keep':[0.1]},
    #{'n_hiddens':n_hiddens,'p_keep':[0.5]},
    {'n_hiddens':n_hiddens}
]
grid = GridSearchCV(estimator=clf, param_grid=param_grid,scoring='roc_auc')
result = grid.fit(X_train_norm,y_train)


