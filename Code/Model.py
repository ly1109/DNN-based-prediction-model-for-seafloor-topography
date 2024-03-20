import tensorflow as tf
from keras.layers import Input, Dense, Concatenate, Conv1D, Flatten, Dropout, Add, Reshape
from keras.models import Model

n = 8
s = 2 ** 4

def dnn(p):
    input = Input(shape=(12,), name='input')
    conv1 = Dense(8 * s, activation=tf.nn.relu)(input)
    conv1 = Dense(16 * s, activation=tf.nn.relu)(conv1)
    conv1 = Dense(32 * s, activation=tf.nn.relu)(conv1)
    conv1 = Dense(64 * s, activation=tf.nn.relu)(conv1)
    conv1 = Dropout(p)(conv1)
    conv1 = Dense(64 * s, activation=tf.nn.relu)(conv1)
    conv1 = Dense(32 * s, activation=tf.nn.relu)(conv1)
    conv1 = Dense(16 * s, activation=tf.nn.relu)(conv1)
    conv1 = Dense(8 * s, activation=tf.nn.relu)(conv1)
    conv1 = Concatenate()([conv1, input])
    conv1 = Dense(1, activation='linear')(conv1)
    model = Model(inputs=input, outputs=conv1)
    return model