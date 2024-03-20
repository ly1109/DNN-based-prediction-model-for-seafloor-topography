import numpy as np
import tensorflow as tf
from Model import dnn
from sklearn.preprocessing import StandardScaler


train_data = np.loadtxt('./Data/Hatteras Plain/Train.csv', delimiter=',')
test_data = np.loadtxt('./Data/Hatteras Plain/Test.csv', delimiter=',')

x = train_data[:, 0:-1]
y = train_data[:, -1]
xt = test_data[:, 0:-1]
yt = test_data[:, -1]


scale1 = StandardScaler().fit(x)
x = scale1.transform(x)
xt = scale1.transform(xt)
scale2 = StandardScaler().fit(y.reshape(-1, 1))
y = scale2.transform(y.reshape(-1, 1))

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


checkpoint_path = "checkpoints/mse_dnn_0.1_4"
ckp = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckp, checkpoint_path, max_to_keep=15)


if ckpt_manager.latest_checkpoint:
    ckp.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')


y_pred1 = model.predict(xt)
y_pred1 = scale2.inverse_transform(y_pred1)
mae1 = np.abs(yt-np.squeeze(y_pred1, axis=1))
mae1 = np.mean(mae1)


y_pred, y_uncertainty = [], []
y1, y2 = [], []
for k in range(500):
    y1 = model(xt, training=True)
    y_pred.append(y1)
    tf.keras.backend.clear_session()


y_pred = np.array(y_pred)
y_pred = np.squeeze(y_pred, 2)

y_pred = scale2.inverse_transform(y_pred)

y_mean = np.mean(y_pred, axis=0)

y_std = np.std(y_pred, axis=0)

np.savetxt('prediction_mean_0.1_4_500.txt', y_pred)
np.savetxt('prediction_std_0.1_4_500.txt', y_std)


