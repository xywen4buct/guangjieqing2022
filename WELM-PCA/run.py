from sklearn.decomposition import KernelPCA
from dataload import get_data
from WELM import MyELM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score,f1_score,roc_auc_score
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

def get_metrics(y_true,y_pred):
    y_pred = np.array(y_pred)
    y_pred_int = np.round(y_pred)
    # print(y_pred)
    # print(y_pred_int)
    # print(y_true)
    recall_s = recall_score(y_true=y_true,y_pred=y_pred_int)
    f1_s = f1_score(y_true=y_true,y_pred=y_pred_int)
    auc = roc_auc_score(y_true=y_true,y_score=y_pred)
    fpr = get_fpr(y_true=y_true,y_pred=y_pred)
    return (recall_s,f1_s,auc,fpr)

def get_fpr(y_true,y_pred):
    y_pred_int = np.round(y_pred)
    y_pred_int = y_pred_int.reshape(-1,)
    fp = np.sum(np.multiply((1-y_true),y_pred_int))
    tn = np.sum(np.multiply((1-y_true),(1-y_pred_int)))
    return fp/(fp+tn)



if __name__ == "__main__":
    data,labels = get_data()
    train_x,train_y,test_x,test_y = split_data(X=data,test_size=0.3,label=labels)
    # gamma = 1.05 * np.std(data) * (len(data)**(-1/5))
    gamma = 1000
    # print(gamma)
    # gamma = 10**2
    # k_pca = KernelPCA(n_components=59,kernel='rbf',gamma=gamma)
    k_pca = KernelPCA(n_components=45,remove_zero_eig=True,kernel='rbf',gamma=gamma)

    train_x_pca = k_pca.fit_transform(X=train_x)
    print(train_x_pca.shape)
    # X_back = k_pca.inverse_transform(X=train_x_pca)
    test_x_pca = k_pca.transform(X=test_x)
    print(test_x_pca.shape)
    # hidden_nodes = train_x_pca.shape[1]
    hidden_nodes = 40
    activation_fun = "kernel"
    random_state = 111
    weighted = True
    # train_x_scal = train_x_pca
    # test_x_scal = test_x_pca
    train_x_scal, scaler = process_data(X=train_x_pca)
    test_x_scal = scaler.transform(X=test_x_pca)
    train_y = train_y.reshape(-1,1)
    test_y = test_y.reshape(-1,1)
    model = MyELM(hidden_nodes=hidden_nodes, activation_fun=activation_fun,
                  random_state=random_state, weighted=weighted, activation_args=None)
    result = model.fit(X=train_x_scal,y=train_y)
    (recall_s,f1_s,auc,fpr) = get_metrics(y_true=train_y,y_pred=result)
    print(
        "train metrics : recall:  %.2f ,  f1_score : %.2f , auc : %.2f, fpr : %.2f" %
        (recall_s, f1_s, auc,fpr)
    )
    pred_result = model.predict(X=test_x_scal)
    (recall_s_t,f1_s_t,auc_t,fpr_t) = get_metrics(y_true=test_y,y_pred=pred_result)
    print(
        "test metrics : recall:  %.2f ,  f1_score : %.2f , auc : %.2f, fpr : %.2f"%
        (recall_s_t,f1_s_t,auc_t,fpr_t)
    )