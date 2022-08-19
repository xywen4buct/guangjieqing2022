from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from numpy.linalg import multi_dot


class MyELMBase(BaseEstimator):
    __metaclass__ = ABCMeta

    def __init__(self, hidden_nodes, activation_fun):
        self.hidden_nodes = hidden_nodes
        self.activation_fun = activation_fun

    @abstractmethod
    def fit(self, X, y=None):
        '''
        :param X: 数据
        :param y: 标签
        :return:
        '''

    @abstractmethod
    def predict(self, X):
        '''
        :param X: 测试数据
        :return:
        '''


class MyELM(MyELMBase, ClassifierMixin):

    def __init__(self, hidden_nodes, activation_fun, random_state, weighted=False, activation_args=None):
        super(MyELM, self).__init__(
            hidden_nodes=hidden_nodes,
            activation_fun=activation_fun,
        )
        self.activation_args = activation_args
        self.random_state = random_state
        self.component = {}
        self.weighted = weighted
        self.activation_fun_im = ActivationFun(activation_name=self.activation_fun). \
            get_activation_fun()
        print(self.activation_fun_im)

    def add_activation_args(self, h_o_a):
        if self.activation_args is not None:
            pass
        return h_o_a

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y doesn't identify")
        h_o_a = self.__compute_hidden_out(X=X)
        h_o_a = self.add_activation_args(h_o_a)
        beta = self.__compute_out_weight(h_o_a, y)
        self.component['beta'] = beta
        result = np.matmul(h_o_a, beta)
        return result

    def __compute_out_weight(self, h_o_a, y):
        # 构造伪逆矩阵
        if self.weighted:
            W_sample = np.zeros(shape=(h_o_a.shape[0],h_o_a.shape[0]))
            index_zero = np.where(y==0)[0]
            index_one = np.where(y==1)[0]
            number_one = np.sum(y)
            number_zero = len(y) - number_one
            W_sample[index_zero,index_zero] = 1.0/number_zero
            W_sample[index_one,index_one] = 0.9612/number_one
            # part_one = np.linalg.inv(np.matmul(np.matmul(h_o_a.T,W_sample),h_o_a))
            part_one = multi_dot([np.linalg.inv(multi_dot([h_o_a.T,W_sample,h_o_a])),h_o_a.T,W_sample])
            # part_one = np.dot(np.dot(np.linalg.inv(np.dot(np.dot(h_o_a.T, W_sample),h_o_a)),h_o_a.T),
            #                   W_sample)
            # part_two = np.matmul(np.matmul(h_o_a.T,W_sample),y)
            return np.dot(part_one,y)
        else:
            p_inv_h = np.dot(np.linalg.inv(np.matmul(h_o_a.T,h_o_a)),h_o_a.T)
            return np.dot(p_inv_h, y)

    def __compute_hidden_out(self, X: np.ndarray):
        np.random.seed(self.random_state)
        F = X.shape[1]
        L = self.hidden_nodes
        W = np.random.uniform(low=-0.5, high=0.5, size=(F, L))
        self.component["weight"] = W
        bias = np.random.uniform(low=-0.5, high=0.5, size=(1,L))
        # bias = np.tile(np.expand_dims(bias,axis=0),(N,1))
        self.component["bias"] = bias
        h_o = np.matmul(X, W) + bias
        h_o_a = self.activation_fun_im(h_o)
        return h_o_a

    def predict(self, X):
        # 前向计算
        h_o_a = self.activation_fun_im(np.matmul(X,self.component["weight"]) + self.component["bias"])
        result = np.matmul(h_o_a,self.component["beta"])
        return result


class ActivationFun(object):
    _sigmoid = (lambda x: 1. / (np.exp(x) + 1.))

    _relu = (lambda x: np.clip(x, 0, x.max()))

    _gaussian = (lambda x: np.exp(-pow(x, 2.0)))

    _activation_fun = {
        "sin": np.sin,
        "tanh": np.tanh,
        "sigmoid": _sigmoid,
        "relu": _relu,
        'kernel':_gaussian
    }

    def __init__(self, activation_name):
        self.activation_name = activation_name

    def get_activation_fun(self):
        print(ActivationFun._activation_fun[self.activation_name])
        return ActivationFun._activation_fun[self.activation_name]
