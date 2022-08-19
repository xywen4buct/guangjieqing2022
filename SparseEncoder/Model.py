import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.metrics import recall_score,f1_score,roc_auc_score
import keras.backend as K
import numpy as np
import os
# from tensorflow.nn import weighted_cross_entropy_with_logits


class Shringking_Layer(tf.keras.layers.Layer):
    def __init__(self,initializer=None):
        super(Shringking_Layer,self).__init__()
        if initializer is None:
            self.initializer = tf.random_uniform_initializer(minval=0.5, maxval=1.0)

    def call(self, inputs, **kwargs):
        self._W = tf.nn.relu(self.w)
        output = tf.multiply(inputs,self._W)
        output = tf.nn.tanh(output)
        return output

    def build(self, input_shape):
        # 负责初始化权重和偏差
        feature_shape = input_shape[1]
        self.w = self.add_weight(
            name='sparse',
            shape=(feature_shape,),
            dtype=tf.float32,
            initializer=self.initializer
        )
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape



class MyModel:
    def __init__(self,input_size,p_hat,pos_rate,alpha,gamma):
        self.pos_rate = pos_rate
        self.alpha = alpha
        self.gamma = gamma
        self.p_hat = tf.convert_to_tensor(p_hat,dtype=tf.float32)
        inputs = layers.Input(shape=[input_size,])
        # self._W_Sparse = tf.Variable(tf.random_normal_initializer()(shape=(input_size,input_size)))
        # self._b_sparse = tf.Variable(tf.zeros_initializer()(shape=input_size))
        #
        # self.sparse_layer = tf.nn.sigmoid(tf.matmul(inputs,self._W_Sparse) + self._b_sparse)
        # [batch_size,input_size]
        # 全连接 达不到效果
        # 做成一个层进行调用
        # self._W = tf.Variable(tf.random_normal_initializer()(shape=(input_size,)))
        # element multiply
        # self._select = tf.multiply(inputs,self._W)
        self._select = Shringking_Layer()(inputs)
        # sparse_layer = layers.Dense(units=input_size,activation='sigmoid',
        #                             kernel_regularizer=tf.keras.regularizers.l1(),
        #                             bias_regularizer=tf.keras.regularizers.l1())(inputs)

        hidden_layer_1 = layers.Dense(units=input_size*2,activation='relu')(self._select)
        norm_layer = layers.BatchNormalization()(hidden_layer_1)
        drop_layer = layers.Dropout(rate=0.2)(norm_layer)
        hidden_layer_2 = layers.Dense(units=input_size,activation='relu')(drop_layer)
        batch_norm = layers.BatchNormalization()(hidden_layer_2)
        outputs = layers.Dense(units=1,activation='sigmoid')(batch_norm)
        self.model = Model(inputs,outputs)
        # self.sparse_model = Model(inputs,sparse_layer)
        self.opt = Adam(learning_rate=1e-4)

    # def get_sparse_weight(self):
    #     return self._W_Sparse
    #
    # def get_sparse_out(self):
    #     return self.sparse_layer

    def loss_classification(self,y_true,y_pred):
        weight = tf.convert_to_tensor((1 - self.pos_rate)/self.pos_rate,dtype=tf.float32)
        # epsilon = tf.keras.backend.epsilon
        epsilon = tf.convert_to_tensor(K.epsilon(),dtype=tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.log(y_pred / (1 - y_pred))
        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred,weight)
        return tf.reduce_mean(cost)

    def focal_loss(self,y_true,y_pred):
        # epsilon = K.epsilon()
        # y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_true = tf.cast(y_true, dtype=tf.float32)

        p_t = tf.multiply(y_true, y_pred) + tf.multiply((tf.ones_like(y_true) - y_true), (tf.ones_like(y_pred) - y_pred))
        ce_loss = -tf.math.log(p_t)
        loss = tf.multiply(tf.pow((tf.ones_like(y_pred) - p_t), self.gamma), ce_loss)
        if self.alpha is not None:
            alpha_t = self.alpha * (y_true) + (1 - self.alpha) * (tf.ones_like(y_true) - y_true)
            loss = tf.multiply(alpha_t, loss)
        return tf.reduce_mean(loss)

    def loss_sparse(self,sparse_out):
        p_hat_den = 1 - self.p_hat
        sparse_den = 1 - sparse_out
        p_hat_den = tf.convert_to_tensor(p_hat_den,dtype=tf.float32)
        sparse_den = tf.convert_to_tensor(sparse_den,dtype=tf.float32)
        # reduce_mean -- [input_size]
        part_one = self.get_logfunc(x=self.p_hat, y=tf.reduce_mean(sparse_out, axis=0))
        part_two = self.get_logfunc(x=p_hat_den, y=sparse_den)
        return tf.reduce_mean(part_one + part_two)

    def get_logfunc(self,x,y):
        return tf.multiply(x,tf.math.log(tf.divide(x,y)))

    def loss_regulization(self,X):
        return tf.reduce_mean(tf.abs(X))

    def train(self,x,epochs):
        # x ,tensor_slice
        losses = []
        recalls = []
        f1_scores = []
        alpha = 1.0
        beta = 1.0
        epsilon = K.epsilon()
        for epoch in range(epochs):
            for i,(train_x,train_y) in enumerate(x):
                # print(train_x.shape)
                # print(train_y.shape)
                with tf.GradientTape() as tape:
                    model_out = self.model(train_x,training=True)
                    print(model_out)
                    W = self.model.layers[1]._W
                    model_out = tf.clip_by_value(model_out,epsilon,1-epsilon)
                    # print(model_out)
                    # print(model_out.shape)
                    # sparse_out = self.sparse_layer
                    # sparse_out = self.sparse_model(train_x,training=False)
                    # print(sparse_out)
                    # loss_cl = self.loss_classification(y_true=train_y,y_pred=model_out)
                    loss_cl = self.focal_loss(y_true=train_y,y_pred=model_out)
                    loss_sp = self.loss_sparse(sparse_out=W)
                    # 0.05
                    loss_re = self.loss_regulization(W)
                    loss = loss_cl + loss_re * alpha + loss_sp * beta

                    # loss = loss_cl
                    losses.append(loss)
                    if (i+1)%10 == 0:
                        # print(loss)
                        recall_ = recall_score(y_true=train_y,y_pred=np.round(model_out))
                        f1_ = f1_score(y_true=train_y,y_pred=np.round(model_out))
                        print("epoch {} , steps {} , cl_loss {},  sp_loss {}, recall {} , f1_score {}".format(
                            epoch,i,loss_cl,loss_sp,recall_,f1_
                        ))
                        # loss_sp
                        # print(self.sparse_model(train_x))
                grads = tape.gradient(loss,self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads,self.model.trainable_variables))
        # self.model.save(filepath="./model.h5")
        # checkpoint = tf.train.Checkpoint(model=self.model,optimizer=self.opt)
        # path_ = "./save_single/ant/{}".format(str(self.alpha)+str(self.gamma))
        # if not os.path.exists(path_):
        #     os.mkdir(path_)
        # save_dir = "./save_single/ant/{}/model.ckpt".format(
        #     str(self.alpha)+str(self.gamma)
        # )
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # path = checkpoint.save(save_dir)
        # print("model save path: {}".format(path))

        return (losses,recalls,f1_scores)

    def predict(self,test_x,test_y):
        pred_val = self.model(test_x,training=False)
        print(pred_val.shape)
        recall_ = recall_score(y_true=test_y,y_pred=np.round(pred_val))
        f1_ = f1_score(y_true=test_y,y_pred=np.round(pred_val))
        auc = roc_auc_score(y_true=test_y,y_score=pred_val.numpy())
        fpr = self.get_fpr(y_true=test_y,y_pred=pred_val)
        print("test: recall {} , f1_score {},  auc {},  fpr {}".format(recall_,f1_,auc,fpr))
        return (pred_val,recall_,f1_,auc,fpr)

    def get_fpr(self,y_true,y_pred):
        y_pred_int = np.round(y_pred)
        y_pred_int = y_pred_int.astype(np.float)
        y_pred_int = y_pred_int
        fp = np.sum(np.multiply((1-y_true),y_pred_int))
        tn = np.sum(np.multiply((1-y_true),(1-y_pred_int)))
        return fp/(fp+tn)
