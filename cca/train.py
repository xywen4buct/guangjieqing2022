from sklearn.cross_decomposition import CCA
from sklearn.metrics import recall_score,f1_score,roc_auc_score
from dataload import get_data
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np


def get_metrics(y_true,y_pred,y_score=None):
    recall = recall_score(y_true=y_true,y_pred=y_pred)
    f1_s = f1_score(y_true=y_true,y_pred=y_pred)
    if y_score is not None:
        auc = roc_auc_score(y_true=y_true,y_score=y_score)
    else:
        auc = roc_auc_score(y_true=y_true,y_score=y_pred)
    fpr = get_fpr(y_true=y_true,y_pred=y_pred)
    return (recall,f1_s,auc,fpr)

def get_fpr(y_true,y_pred):
    fp = np.sum(np.multiply((1 - y_true), y_pred))
    tn = np.sum(np.multiply((1 - y_true), (1 - y_pred)))
    return fp / (fp + tn)

def scale(data,style):
    if style == "mm":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scale_data = scaler.fit_transform(data)
    return scale_data

data_source,label_source = get_data(db_name="kafka")
data_target,label_target = get_data(db_name="kylin")

data_source = scale(data=data_source,style="std")
data_target = scale(data=data_target,style="std")

# 构建相同样本量的数据
ratio = len(data_source)/len(data_target)

# source多，target少
if ratio > 1.0:
    ratio_ = len(data_target)/len(data_source)
    data_source,tmp,label_source,tmp_label = train_test_split(
        data_source,label_source,test_size=1-ratio_,random_state=111,stratify=label_source
    )
    # 去除数据多了，进行补充
    if len(data_source)<len(data_target):
        num = len(data_target) - len(data_source)
        data_source = np.vstack([data_source,tmp[:num,:]])
        label_source = np.concatenate([label_source,tmp_label[:num]])
    # 去除数据后数据仍然较多
    elif len(data_target)<len(data_source):
        data_source = data_source[:len(data_target)]
        label_source = label_source[:len(label_target)]
# source少，target多
elif ratio < 1.0:
    data_target, tmp, label_target, tmp_label = train_test_split(
        data_target, label_target, test_size=1 - ratio, random_state=111, stratify=label_target
    )
    # 如果data_target较少了,将data_target再补充
    if len(data_target) < len(data_source):
        num = len(data_source) - len(data_target)
        data_target = np.vstack([data_target,tmp[:num,:]])
        label_target = np.concatenate([label_target, tmp_label[:num]])
    # 如果data_source少数选择后的data_target,对data_target进行删减
    elif len(data_source) < len(data_target):
        data_target = data_target[:len(data_source),:]
        label_target = label_target[:len(label_source)]


cca = CCA(n_components=50,scale=False)

data_source_cca,data_target_cca = cca.fit_transform(X=data_source,y=data_target)

weights = "uniform"

# 是否需要归一化，最大和最小值不在[0,1]内
# print(np.min(data_target_cca))
# print(np.max(data_target_cca))
# print(np.min(data_source_cca))
# print(np.max(data_source_cca))

# weights = "distance"
# clf = KNeighborsClassifier(n_neighbors=5,weights=weights,)
# clf = RandomForestClassifier(n_estimators=50,class_weight={0:0.2,1:0.8})
# 采用Balanced的分类器更好一些
# clf = BalancedBaggingClassifier(n_estimators=50,random_state=111)
clf = BalancedRandomForestClassifier(n_estimators=50,random_state=111)
# data_source_cca = scale(data_source_cca,"mm")
# data_target_cca = scale(data_target_cca,"mm")

# 参数搜索
# params = {
#     'gamma':[1e-1,1.,10.,100.,1e3],
#     'C':[1.,10.,100.,1e3]
# }

# clf = SVC(kernel="rbf",class_weight={0:0.2,1:0.8},C=10.,gamma=10.,probability=True)

# clf = GridSearchCV(estimator=svc,param_grid=params,cv=5)

clf.fit(data_source_cca,label_source)
# print(clf.best_params_)
# print(clf.best_estimator_)
# estimator = clf.best_estimator_
# res = estimator.predict(data_target_cca)
# print(res)
res = clf.predict(data_target_cca)

# prob_res = clf.predict_proba(data_target_cca)[:,1]

# print(prob_res)
recall,f1_s,auc,fpr = get_metrics(y_true=label_target,y_pred=res)
#
print(recall)
print(f1_s)
print(auc)
print(fpr)