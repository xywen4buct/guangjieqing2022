import tensorflow as tf
from tensorflow.keras import layers
import keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import recall_score,f1_score,roc_auc_score
import numpy as np

class Model_Without_Select:
    def __init__(self,input_size,p_hat,pos_rate,alpha,gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.pos_rate = pos_rate
        self.p_hat = tf.convert_to_tensor(p_hat,dtype=tf.float32)
        inputs = layers.Input(shape=[input_size,])
        hidden_layer_1 = layers.Dense(units=input_size*2,activation='relu')(inputs)
        norm_layer = layers.BatchNormalization()(hidden_layer_1)
        drop_layer = layers.Dropout(rate=0.2)(norm_layer)
        hidden_layer_2 = layers.Dense(units=input_size,activation='relu')(drop_layer)
        batch_norm = layers.BatchNormalization()(hidden_layer_2)
        outputs = layers.Dense(units=1,activation='sigmoid')(batch_norm)
        self.model = Model(inputs,outputs)
        self.opt = Adam(learning_rate=1e-4)

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

    def train(self,x,epochs):
        # x ,tensor_slice
        losses = []
        recalls = []
        f1_scores = []
        epsilon = K.epsilon()
        for epoch in range(epochs):
            for i,(train_x,train_y) in enumerate(x):
                # print(train_x.shape)
                # print(train_y.shape)
                with tf.GradientTape() as tape:
                    model_out = self.model(train_x,training=True)
                    model_out = tf.clip_by_value(model_out, epsilon, 1 - epsilon)
                    # print(model_out.shape)
                    # sparse_out = self.sparse_layer
                    # sparse_out = self.sparse_model(train_x,training=False)
                    # print(sparse_out)
                    # loss_cl = self.loss_classification(y_true=train_y,y_pred=model_out)
                    loss_cl = self.focal_loss(y_true=train_y, y_pred=model_out)
                    # loss_sp = self.loss_sparse(sparse_out=W)
                    # 0.05
                    loss = loss_cl
                    losses.append(loss)
                    if (i+1)%10 == 0:
                        # print(loss)
                        recall_ = recall_score(y_true=train_y,y_pred=np.round(model_out))
                        f1_ = f1_score(y_true=train_y,y_pred=np.round(model_out))
                        print("epoch {} , steps {} , cl_loss {},  recall {} , f1_score {}".format(
                            epoch,i,loss_cl,recall_,f1_
                        ))
                        # loss_sp
                        # print(self.sparse_model(train_x))
                grads = tape.gradient(loss,self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads,self.model.trainable_variables))
        # self.model.save(filepath="./model.h5")
        checkpoint = tf.train.Checkpoint(model=self.model,optimizer=self.opt)
        path = checkpoint.save('./save_no_select/tomcat/model.ckpt')
        print("model save path: {}".format(path))
        return (losses,recalls,f1_scores)

    def predict(self, test_x, test_y):
        pred_val = self.model(test_x, training=False)
        recall_ = recall_score(y_true=test_y, y_pred=np.round(pred_val))
        f1_ = f1_score(y_true=test_y, y_pred=np.round(pred_val))
        auc = roc_auc_score(y_true=test_y, y_score=pred_val.numpy())
        fpr = self.get_fpr(y_true=test_y, y_pred=pred_val)
        print("test: recall {} , f1_score {},  auc {},  fpr {}".format(recall_, f1_, auc, fpr))
        return (pred_val, recall_, f1_, auc, fpr)

    def get_fpr(self,y_true,y_pred):
        y_pred_int = np.round(y_pred)
        y_pred_int = y_pred_int
        fp = np.sum(np.multiply((1-y_true),y_pred_int))
        tn = np.sum(np.multiply((1-y_true),(1-y_pred_int)))
        return fp/(fp+tn)