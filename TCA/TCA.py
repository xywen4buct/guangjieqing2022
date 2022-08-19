import numpy as np
from sklearn.metrics import recall_score,f1_score,roc_auc_score,pairwise
from scipy.linalg import eig
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from dataload import get_data
from normalization_utils import select_normalization,n3_normal,n4_normal
from sklearn.preprocessing import MinMaxScaler,StandardScaler

"""
求解核矩阵K
    ker:求解核函数的方法
    X1:源域数据的特征矩阵
    X2:目标域数据的特征矩阵
    gamma:当核函数方法选择rbf时，的参数
"""
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker=='primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K

class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1., gamma=1.):
        """
        :param kernel_type:
        :param dim:
        :param lamb:
        :param gamma:
        """
        self.kernel_type = kernel_type  #选用核函数的类型
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        """
        :param Xs: 源域的特征矩阵 （样本数x特征数）
        :param Xt: 目标域的特征矩阵 （样本数x特征数）
        :return: 经过TCA变换后的Xs_new,Xt_new
        """
        # 构造MMD矩阵
        # 维度变为 feature*(nt + ns)
        X = np.hstack((Xs.T, Xt.T))
        # 求范式,根据范式归一化  ？？？？
        # X = X/np.linalg.norm(X, axis=0)
        # m--特征数  n--总样本数
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        # 构造MMD矩阵 L
        # 维度为 (ns+nt)*1
        e = np.vstack((1 / ns*np.ones((ns, 1)), -1 / nt*np.ones((nt, 1))))
        # (ns+nt)*(ns+nt)
        M = e * e.T
        # 归一化 ????
        # M = M / np.linalg.norm(M, 'fro')
        # 构造中心矩阵H=E-1/N * I  (ns+nt)*(ns+nt)
        H = np.eye(n) - (1/n)*np.ones((n, n))
        # 构造核函数矩阵K,直接根据X构造矩阵  (ns+nt)*(ns+nt)
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        # primal 返回X (m * n)
        n_eye = m if self.kernel_type == 'primal' else n

        # 注意核函数K就是后边的X特征，只不过用核函数的形式表示了
        a = np.linalg.multi_dot([K, M, K]) + self.lamb * np.eye(n_eye)#XMX_T+lamb*I
        a_reverse = np.linalg.inv(a)
        b = np.linalg.multi_dot([K, H, K])#XHX_T
        solve_matrix = np.matmul(a_reverse,b)
        w,V = eig(a=solve_matrix)
        ind = np.argsort(w)#argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到ind
        A = V[:, ind[:self.dim]]#取前dim个特征向量得到变换矩阵A，按照特征向量的大小排列好,
        Z = np.dot(A.T, K)#将数据特征*映射A
        Z /= np.linalg.norm(Z, axis=0)#单位向量化

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T#得到源域特征和目标域特征
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        """
        Transform Xs and Xt , then make p r edi c tion s on ta rg e t using 1NN
        param Xs : ns ∗ n_feature , source feature
        param Ys : ns ∗ 1 , source label
        param Xt : nt ∗ n_feature , target feature
        param Yt : nt ∗ 1 , target label
        return : Accuracy and predicted_labels on the target domain
        """
        Xs_new, Xt_new = self.fit(Xs, Xt)#经过TCA映射
        # clf = KNeighborsClassifier(n_neighbors=1) #k近邻分类器，无监督学习
        clf = RandomForestClassifier(min_samples_split=10,min_samples_leaf=5,
                                     class_weight={0:0.2,1:0.8},random_state=111)
        clf.fit(Xs_new, Ys.ravel())#训练源域数据
        # 然后直接用于目标域的测试
        y_pred = clf.predict(Xs_new)
        y_pred_score = clf.predict_proba(Xs_new)
        y_pred_score = np.max(y_pred_score,axis=1)
        recall = recall_score(y_true=Ys,y_pred=y_pred)
        f1_s = f1_score(y_true=Ys,y_pred=y_pred)
        auc = roc_auc_score(y_true=Ys,y_score=y_pred_score.ravel())
        # acc = sklearn.metrics.accuracy_score(Ys, y_pred)
        print("recall : {}, f1_score : {}, auc : {}".format(recall,f1_s,auc))
        y_pred = clf.predict(Xt_new)
        y_pred_score = clf.predict_proba(Xt_new)
        y_pred_score = np.max(y_pred_score, axis=1)
        recall = recall_score(y_true=Yt, y_pred=y_pred)
        f1_s = f1_score(y_true=Yt, y_pred=y_pred)
        auc = roc_auc_score(y_true=Yt, y_score=y_pred_score.ravel())
        # acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return (recall,f1_s,auc,y_pred)
# dslr_SURF_L10.mat -> webcam_SURF_L10.mat
# acc: 0.688135593220339
if __name__ == '__main__':
    """
    domains = ['amazon_SURF_L10.mat', 'Caltech10_SURF_L10.mat', 'dslr_SURF_L10.mat', 'webcam_SURF_L10.mat']
    src = 'SURF/' + "Caltech10_SURF_L10.mat"
    tar = 'SURF/' + "amazon_SURF_L10.mat"
    # dict数据，fts为数据，labels为标签
    src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    # print(src_domain)
    # print(tar_domain)
    # 获取数据
    Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']
    print(Xs.shape)
    print(Xt.shape)
    # lamb为调节因子， 优化目标中对W.T * W的约束
    tca = TCA(kernel_type='rbf', dim=50, lamb=0.5, gamma=10.)
    # Xs_new,Xt_new = tca.fit(Xs,Xt)
    # print(Xs_new)
    # print(Xt_new)
    acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
    print(src, "->", tar)
    print("acc:", acc)
    """
    #k近邻设置为3
    # 0.6714158504007124
    # SURF / Caltech10_SURF_L10.mat -> SURF / amazon_SURF_L10.mat
    # acc: 0.40292275574112735
    # for i in range(4):
    #     for j in range(4):
    #         if i != j:
    #             src= 'SURF/' + domains[i]
    #             tar = 'SURF/' + domains[j]
    #             src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    #             # 获取数据
    #             Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']
    #
    #             tca = TCA(kernel_type='rbf', dim=30, lamb=1, gamma=1)
    #             acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
    #             print(domains[i], "->", domains[j])
    #             print("acc:",acc)

    data_source,label_source = get_data(db_name="kafka")
    data_target,label_target = get_data(db_name="kylin")
    _,data_source_select,_,label_source_select = train_test_split(data_source,
                                                                   label_source,
                                                                   test_size=0.1,
                                                                   random_state=111,
                                                                   stratify=label_source)
    _,data_target_select,_,label_target_select = train_test_split(data_target,
                                                                  label_target,
                                                                  test_size=0.1,
                                                                  random_state=111,
                                                                  stratify=label_target)
    print("select normaliztion.....")
    # 距离计算耗时
    # normal_res = "N1"
    normal_res = select_normalization(source_data=data_source_select,target_data=data_target_select)
    if normal_res == "N1":
        scale = MinMaxScaler()
        data_s_scale = scale.fit_transform(data_source_select)
        data_t_scale = scale.transform(data_target_select)
    elif normal_res == "N2":
        scale = StandardScaler()
        data_s_scale = scale.fit_transform(data_source_select)
        data_t_scale = scale.transform(data_target_select)
    elif normal_res == "N3":
        data_s_scale = n3_normal(source_data=data_source_select,normal_data=data_source_select)
        data_t_scale = n3_normal(source_data=data_source_select,normal_data=data_target_select)
    elif normal_res == "N4":
        data_s_scale = n4_normal(target_data=data_target_select,normal_data=data_source_select)
        data_t_scale = n4_normal(target_data=data_target_select,normal_data=data_target_select)
    else:
        data_s_scale = data_source_select
        data_t_scale = data_target_select
    print("normalization :\t",normal_res)
    # 参数设置，文章未提，主要参数设置有gamma和降维到的维度dim
    print("model start .....")
    kernel_type = "rbf"
    dim = 100
    lamb = 1.
    # gamma自动设置为 1/features
    gamma = 0.5
    # 分类器，文章中使用的是逻辑回归
    tca = TCA(kernel_type=kernel_type, dim=dim, lamb=lamb, gamma=gamma)
    Xs_new,Xt_new = tca.fit(data_s_scale,data_t_scale)
    print(Xs_new)
    print(Xt_new)
    # recall,f1_s,auc,ypre = tca.fit_predict(data_s_scale,label_source ,
    #                                        data_t_scale,label_target)
    # print("kafka", "->", "kylin")
    # print("recall:\t",recall)
    # print("f1_score:\t",f1_s)
    # print("auc:\t",auc)
