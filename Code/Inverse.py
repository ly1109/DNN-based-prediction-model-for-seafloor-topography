import numpy as np
from Model import dnn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import itertools
import time
from keras.callbacks import ReduceLROnPlateau

train_data = np.loadtxt('./Data/Hatteras Plain/Train.csv', delimiter=',')
test_data = np.loadtxt('./Data/Hatteras Plain/Test.csv', delimiter=',')

x = train_data[:, 0:-1]
y = train_data[:, -1]
xte = test_data[:, 0:-1]
yte = test_data[:, -1]

scale1 = StandardScaler().fit(x)
x = scale1.transform(x)
xte = scale1.transform(xte)
scale2 = StandardScaler().fit(y.reshape(-1, 1))
y = scale2.transform(y.reshape(-1, 1))
yte = scale2.transform(yte.reshape(-1, 1))

p = 0.1
model = dnn(p)
model.summary()

optimizer = tf.optimizers.Adam(learning_rate=1e-3,
                               beta_1=0.9,
                               beta_2=0.999,
                               epsilon=1e-07,
                               amsgrad=False,
                               clipvalue=0.5,
                               clipnorm=1)


def loss_func(real, fake):
    mse = tf.reduce_mean(tf.square(real - fake))
    return mse


def metric_func(real, fake):
    mae = tf.reduce_mean(tf.abs(scale2.inverse_transform(real) - scale2.inverse_transform(fake)))
    mse = tf.reduce_mean(tf.square(scale2.inverse_transform(real) - scale2.inverse_transform(fake)))
    return mae, mse


checkpoint_path = "checkpoints/mse_dnn_0.1_1"
ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=15)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

EPOCHS = 500
batch_size = 240
all_loss = []
val_loss = []
for epoch in range(EPOCHS):
    start = time.time()
    epoch_losses = []
    epoch_time = []

    for i in range(int(x.shape[0] / batch_size)):
        begin, end = i * batch_size, (i + 1) * batch_size
        xt, yt = x[begin:end, :], y[begin:end]
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(xt, training=True)
            loss = loss_func(yt, y_pred)
            metric1, metric2 = metric_func(yt, y_pred)
            epoch_losses.append((loss.numpy(), metric1.numpy(), metric2.numpy()))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print('Epoch {:03d} | ET {:.2f} min'.format(i + 1, (time.time() - start) / 60), end=' ')
        print(epoch_losses[-1])

    if (x.shape[0] % batch_size) != 0:
        i = i+1
        xt, yt = x[end:-1, :], y[end:-1]
        with tf.GradientTape(persistent=True) as tape:
            y_pred = model(xt, training=True)
            loss = loss_func(yt, y_pred)
            metric1, metric2 = metric_func(yt, y_pred)
            epoch_losses.append((loss.numpy(), metric1.numpy(), metric2.numpy()))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print('Epoch {:03d} | ET {:.2f} min'.format(i + 1, (time.time() - start) / 60), end=' ')
        print(epoch_losses[-1])

    all_loss.append(epoch_losses)

    print('Epoch={},Loss:{},Metric:{},Metric:{}'.format(epoch + 1, *list(np.mean(all_loss[-1], axis=0))))

    yp = model(xte)
    metric3, metric4 = metric_func(yte, yp)
    print(metric3.numpy(), metric4.numpy())

    if (epoch + 1) % 100 == 0:  # & (epoch + 1) <= 20
        optimizer.lr.assign(optimizer.lr / 2.0)

    if (epoch + 1) % 100 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

    shuffle_ix = np.random.permutation(np.arange(x.shape[0]))
    x = x[shuffle_ix, :]
    y = y[shuffle_ix]

    tf.keras.backend.clear_session()


loss1 = [item[0] for item in itertools.chain(*all_loss)]
loss2 = [item[1] for item in itertools.chain(*all_loss)]
loss3 = [item[2] for item in itertools.chain(*all_loss)]
np.savetxt('mse_loss_dnn_0.1_1.txt', loss1)
np.savetxt('mse_mae_dnn_0.1_1.txt', loss2)
np.savetxt('mse_mse_dnn_0.1_1.txt', loss3)
