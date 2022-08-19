from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dataload import get_data
from Model_Naive import Model_Without_Select
import tensorflow as tf
import numpy as np

def split_data(X,test_size,label,random_state=111):
    train_X,test_X,train_Y,test_Y = train_test_split(X,label,test_size=test_size,
                                                         stratify=label,random_state=random_state)
    return (train_X,train_Y,test_X,test_Y)

def process_data(X):
    '''
    :param X: 应当传入训练集
    :return:
    '''
    scale = MinMaxScaler()
    result = scale.fit_transform(X=X)
    return (result,scale)

if __name__ == "__main__":
    # model_ = MyModel(input_size=59, p_hat=0.05, pos_rate=0.2,alpha=0.6,gamma=0.5)
    # model = model_.model
    # opt = model_.opt
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    # checkpoint.restore(tf.train.latest_checkpoint('./save/tomcat/0.60.5'))
    # thresh = model.weights[0]
    # thresh = tf.nn.relu(thresh).numpy()
    data, label = get_data()
    label = label[:, np.newaxis]
    label = label.astype(np.float32)
    train_X, train_Y, test_X, test_Y = split_data(X=data, test_size=0.3, label=label)
    train_x_scale, scaler = process_data(X=train_X)
    test_x_scale = scaler.transform(X=test_X)
    # train_x_scale = np.multiply(train_x_scale,thresh)
    # test_x_scale = np.multiply(test_x_scale,thresh)
    # print(train_x_scale)
    train_data = tf.data.Dataset.from_tensor_slices((train_x_scale, train_Y)).shuffle(buffer_size=64).batch(
        batch_size=64
    )
    model_w_s = Model_Without_Select(input_size=train_x_scale.shape[1],p_hat=0.01,pos_rate=0.2,alpha=0.8,gamma=0.5)
    (losses,recalls,f1_scores) = model_w_s.train(x=train_data,epochs=15)
    # print(model.model.weights)
    # model = model_w_s.model
    # opt = model_w_s.opt
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
    # checkpoint.restore(tf.train.latest_checkpoint('./save_no_select/kafka'))
    # print(losses)
    # print(recalls)
    # print(f1_scores)
    result = model_w_s.predict(test_x_scale,test_Y)
    print("recall:\t",result[1])
    print("f1_score:\t",result[2])
    print("auc:\t",result[3])
    print("fpr:\t",result[4])
    # print(result)