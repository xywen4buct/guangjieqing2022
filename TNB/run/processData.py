from utils.processdata import DiscreteByEntropy
import pandas as pd
import numpy as np
from dataload.dataloader import get_data,get_data_file


# dataset = get_data_file(file_path="../data/KC1.csv")
dataset,labels = get_data(db_name="kafka")
# labels = dataset.iloc[:,-1]

features = dataset.shape[1]-1

dis = DiscreteByEntropy(group=500,threshold=0.5)

labels = labels.reshape(-1,1)

data_all = np.hstack([dataset,labels])

print(data_all.shape)

dataFrame = pd.DataFrame(data=data_all,columns=[i+1 for i in range(data_all.shape[1])])

# 对每一列数据进行离散化
last_col = dataFrame.shape[1]
# 计算化数据保存文件


for i in range(dataFrame.shape[0]):
    tmp_container = dataFrame.iloc[:,[i+1,dataFrame.shape[1]-1]].values
    # print(tmp_container)
    # break
    dis.train(tmp_container)
    print(dis.result)
    break
