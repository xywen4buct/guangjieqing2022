from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from dataload import get_data
from Model import MyModel
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
    data,label = get_data()
    label = label[:,np.newaxis]
    label = label.astype(np.float32)
    train_X,train_Y,test_X,test_Y = split_data(X=data,test_size=0.3,label=label)

    train_x_scale,scaler = process_data(X=train_X)
    test_x_scale = scaler.transform(X=test_X)
    train_data = tf.data.Dataset.from_tensor_slices((train_x_scale,train_Y)).shuffle(buffer_size=64).batch(
        batch_size=64
    )
    '''
    alphas = [0.2,0.4,0.6,0.8]
    gammas = [0.,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
    for a in alphas:
        for g in gammas:
            print("a= {}, g={} ".format(a,g))
            tmp = MyModel(input_size=train_x_scale.shape[1],p_hat=0.01,pos_rate=0.2,alpha=a,gamma=g)
            (losses, recalls, f1_scores) = tmp.train(x=train_data, epochs=100)
            pred_val,recall_,f1_,auc,fpr = tmp.predict(test_x_scale, test_Y)
            # 记录结果画热力图
            with open("./save/result/tomcat/{}.txt".format(str(a)+str(g)),'w') as f:
                f.write("recall: " + str(recall_) + "\n")
                f.write("f1_score: " + str(f1_) + "\n")
                f.write("auc: " + str(auc) + "\n")
                f.write("fpr: " + str(fpr) + "\n")
                f.flush()
                f.close()
    '''
    # print(test_Y.shape)
    model = MyModel(input_size=train_x_scale.shape[1],p_hat=0.01,pos_rate=0.2,alpha=0.8,gamma=0.5)
    (losses,recalls,f1_scores) = model.train(x=train_data,epochs=40)
    # print(model.model.weights)
    # print(losses)
    # print(recalls)
    # print(f1_scores)
    result = model.predict(test_x_scale,test_Y)
    print(result)