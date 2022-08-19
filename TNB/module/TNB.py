import numpy as np
from dataload.dataloader import get_data
from collections import defaultdict
# from numpy.linalg import multi_dot
from utils.metrics import get_metrics


source_dataset,source_labels = get_data(db_name="kafka")
target_dataset,target_labels = get_data(db_name="kylin")

n_feature = source_dataset.shape[1]
n_source_sample = source_dataset.shape[0]

sim_matrix = np.zeros(shape=(n_source_sample,n_feature))

for i in range(n_feature):
    min_v = np.min(target_dataset[:,i])
    max_v = np.max(target_dataset[:,i])
    for j in range(n_source_sample):
        if source_dataset[j][i]<= max_v and source_dataset[j][i]>=min_v:
            sim_matrix[j][i] = 1
        else:
            sim_matrix[j][i] = 0

# 求整体相似向量
sim_vec = np.sum(sim_matrix,axis=1)

weight_vec = np.zeros_like(sim_vec)

for i in range(len(sim_vec)):
    weight_vec[i] = sim_vec[i]/((59-sim_vec[i]+1)**2)

print(weight_vec)


# 计算目标域的先验概率
weight_c_1 = np.sum(
    np.multiply(weight_vec,np.equal(source_labels,1.).astype(np.int))
)+1
weight_all = np.sum(weight_vec)+2

weight_c_0 = np.sum(
    np.multiply(weight_vec,np.equal(source_labels,0.).astype(np.int))
)

# 目标域的先验概率
p_c_1 = weight_c_1/weight_all
p_c_0 = weight_c_0/weight_all


# 获取离散化后的结果
discrete_data_source = np.array()
discrete_data_target = np.array()

# 根据离散化的结果进行计算各个属性的条件概率
condition_probs = defaultdict(dict)

for i in range(n_feature):
    unique_val = np.unique(discrete_data_target[:,i])
    condition_prob_f = dict()
    for val in unique_val:
        condition_prob_f[str(val)+"_0"] = np.sum(
            np.multiply(
                (np.multiply(weight_vec,np.equal(discrete_data_source[:,i],val).astype(np.int)))),
                np.equal(source_labels,0.).astype(np.int)
        )
        condition_prob_f[str(val)+"_1"] = np.sum(
            np.multiply(
                (np.multiply(weight_vec,np.equal(discrete_data_source[:,i],val).astype(np.int))),
                np.equal(source_labels,1.).astype(np.int)
            )
        )
    condition_probs[i+1] = condition_prob_f

# 分类

result = []
for i in range(len(target_labels)):
    tmp_res_0 = 1.
    tmp_res_1 = 1.
    for j in range(n_feature):
        val = discrete_data_target[i][j]
        tmp_res_0 *= condition_probs[i+1][str(val)+"_0"]
        tmp_res_1 *= condition_probs[i+1][str(val)+"_1"]
    if tmp_res_1 > tmp_res_0:
        result.append(1)
    else:
        result.append(0)
recall,f1_s,auc,fpr = get_metrics(y_true=target_labels,y_pred=np.array(result))
# print(result)
print(recall)
print(f1_s)
print(auc)
print(fpr)