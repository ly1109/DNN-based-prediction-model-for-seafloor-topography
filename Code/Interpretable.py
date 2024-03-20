import numpy as np
import tensorflow as tf
from Model import dnn
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt


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
yt = scale2.transform(yt.reshape(-1, 1))

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

checkpoint_path = "mse_dnn_0.1"
ckp = tf.train.Checkpoint(model=model, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckp, checkpoint_path, max_to_keep=15)


if ckpt_manager.latest_checkpoint:
    ckp.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

feature = ['LON', 'LAT', 'GA', 'RGA', 'VGG', 'RVGG', 'EVD', 'NVD', 'IGA', 'ST', 'MD', 'MA']

x_summary = shap.sample(x, 50)
xt_summary = shap.sample(xt, 2 * xt.shape[1] + 2048)

explainer = shap.KernelExplainer(model, x_summary)
shap_value = explainer(xt_summary)

value = []
for k in range(20):
    explainer = shap.KernelExplainer(model, x_summary)
    shap_value = explainer(xt_summary)
    value.append(shap_value.values)
    tf.keras.backend.clear_session()

value = np.array(value)
value = np.squeeze(value, 3)
np.save(file='hatteras_plain_shap_value_0.1.npy', arr=value)


value_mean = np.mean(value, axis=0)
value_std = np.std(value, axis=0)

shap.summary_plot(value_mean, xt_summary, feature_names=feature, plot_type='dot')
shap.summary_plot(value_mean, xt_summary, feature_names=feature, plot_type='bar')
