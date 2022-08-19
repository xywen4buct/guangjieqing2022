import numpy as np

def select_normalization(source_data,target_data):
    '''
    判定规则
    :param source_data:
    :param target_data:
    :return:
    '''
    s_d = compute_distance(data=source_data)
    t_d = compute_distance(data=target_data)
    s_d = np.array(s_d)
    t_d = np.array(t_d)
    # 获取距离的均值、标准差、中位数、最小值、最大值、距离数量
    s_d_num = len(s_d)
    t_d_num = len(t_d)
    s_d_mean = np.mean(s_d)
    t_d_mean = np.mean(t_d)
    s_d_std = np.std(s_d)
    t_d_std = np.std(t_d)
    s_d_media = s_d[int((len(s_d)-1)/2)] if (len(s_d)-1)%2 == 0. else \
        (s_d[int((len(s_d)-2)/2)]+s_d[int(len(s_d)/2)])/2
    t_d_media = t_d[int((len(t_d)-1)/2)] if (len(t_d)-1)%2 == 0. else \
        (t_d[int((len(t_d)-2)/2)]+t_d[int(len(t_d)/2)])/2
    s_d_max = np.max(s_d)
    t_d_max = np.max(t_d)
    s_d_min = np.min(s_d)
    t_d_min = np.min(t_d)
    s_d_info = dict((("mean",s_d_mean),("std",s_d_std),("insNum",s_d_num),("max",s_d_max),
                     ("min",s_d_min),("media",s_d_media)))
    t_d_info = dict((("mean", t_d_mean), ("std", t_d_std), ("insNum", t_d_num),
                     ("max", t_d_max), ("min", t_d_min), ("media", t_d_media)))
    # 判断规则
    if s_d_info["mean"]*0.9 <= t_d_info["mean"] and t_d_info["mean"]<= s_d_info["mean"]*1.1 and \
            s_d_info["std"]*0.9 <= t_d_info["std"] and t_d_info["std"]<= s_d_info["std"]*1.1:
        return "NoN"
    if (s_d_info["insNum"]*1.6<t_d_info["insNum"] or s_d_info["insNum"]*0.4 > t_d_info["insNum"]) and \
            (s_d_info["min"]*1.6<t_d_info["min"] or s_d_info["min"]*0.4>t_d_info["min"]) and \
            (s_d_info["max"]*1.6<t_d_info["max"] or s_d_info["max"]*0.4>t_d_info["max"]):
        return "N1"
    if (s_d_info["std"]*1.6<t_d_info["std"] and s_d_info["insNum"]*1.1>=t_d_info["insNum"]) or \
        (s_d_info["std"]*0.4>t_d_info["std"] and s_d_info["insNum"]*1.1<=t_d_info["insNum"]):
        return "N3"

    if (s_d_info["std"]*1.6<t_d_info["std"] and s_d_info["insNum"]*1.6<t_d_info["insNum"])\
            or (s_d_info["std"]*0.4>t_d_info["std"] and s_d_info["insNum"]*0.4>t_d_info["insNum"]):
        return "N4"
    return "N2"

def compute_distance(data):
    '''
    :param data: 二维矩阵
    :return:
    '''
    distance_order = []
    '''
    # 计算所有数据对的距离
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            dis = np.sqrt(np.sum(np.power((data[i,:] - data[j,:]),2)))
            distance_order.append(dis)
    '''

    s_e = np.expand_dims(data, axis=1)
    s = np.tile(s_e, [1, data.shape[0], 1])
    s_e_ = np.expand_dims(data, axis=0)
    s_ = np.tile(s_e_, [data.shape[0], 1, 1])
    r = s - s_
    r = r ** 2
    distance_all = np.sqrt(np.sum(r, axis=2))
    distance_all = np.triu(distance_all)
    index_x, index_y = np.where(distance_all != 0)
    for i, j in zip(index_x, index_y):
        distance_order.append(distance_all[i][j])

    return distance_order

def n3_normal(source_data,normal_data):
    std_f = np.std(source_data)
    mean_f = np.mean(source_data)
    return (normal_data-mean_f)/std_f

def n4_normal(target_data,normal_data):
    std_f = np.std(target_data)
    mean_f = np.mean(target_data)
    return (normal_data - mean_f)/std_f


if __name__ == "__main__":
    a = np.array([[1,2,3],[4,5,6],[3,2,1],[5,6,3]])
    b = np.array([[1,1,2],[2,1,3],[3,2,1],[1,2,5]])
    dis = compute_distance(a)
    print(dis)
    # dis = select_normalization(a,b)
    # print(dis)
    pass