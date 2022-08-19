import tensorflow as tf
from Model import MyModel
from dataload import get_data
import numpy as np

# data,label = get_data()
model_ = MyModel(input_size=59, p_hat=0.05, pos_rate=0.2,alpha=0.8,gamma=0.5)
model = model_.model
opt = model_.opt
checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
checkpoint.restore(tf.train.latest_checkpoint('./save/kafka/0.60.5'))
shrinking = model.weights[0].numpy()
print(shrinking)
shrinking = np.where(shrinking<0,0,shrinking)
print(shrinking)
# w = model.weights[0].numpy()
# b = model.weights[1].numpy()
# res = np.matmul(data,w) + b
# print(res)
