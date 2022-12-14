import math

import numpy as np


class DiscreteByEntropy:
    def __init__(self, group, threshold):
        self.maxGroup = group  # 最大分组数
        self.minInfoThreshold = threshold  # 停止划分的最小熵
        self.result = dict()  # 保存划分结果

    # 准备数据
    def loadData(self):
        data = np.array(
            [
                [56, 1], [87, 1], [129, 0], [23, 0], [342, 1],
                [641, 1], [63, 0], [2764, 1], [2323, 0], [453, 1],
                [10, 1], [9, 0], [88, 1], [222, 0], [97, 0],
            ]
        )
        return data

    def calEntropy(self, data):
        numData = len(data)
        labelCounts = {}
        for feature in data:
            # 获得标签
            oneLabel = feature[-1]
            # 如果标签步骤新定义的字典里则创建该标签
            labelCounts.setdefault(oneLabel, 0)
            # 该类标签下含有数据的个数
            labelCounts[oneLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            # 同类标签出现的概率
            prob = float(labelCounts[key]) / numData
            # 以2为底求对数
            shannonEnt -= prob * math.log(prob, 2)
        return shannonEnt

    def split(self, data):
        # inf为正无穷大
        minEntropy = np.inf
        # 记录最终分割索引
        index = -1
        # 按照第一列对数据进行排序
        sortData = data[np.argsort(data[:, 0])]
        # 初始化最终分割数据后的熵
        lastE1, lastE2 = -1, -1
        # 返回的数据结构，包含数据和对应的熵
        s1 = dict()
        s2 = dict()
        for i in range(len(sortData)):
            # 分割数据集 对所有可能的分割点进行迭代 data中带有label
            splitData1, splitData2 = sortData[: i + 1], sortData[i + 1:]
            entropy1, entropy2 = (
                self.calEntropy(splitData1),
                self.calEntropy(splitData2),
            )  # 计算信息熵
            entropy = entropy1 * len(splitData1) / len(sortData) + \
                      entropy2 * len(splitData2) / len(sortData)
            # 如果调和平均熵小于最小值
            if entropy < minEntropy:
                minEntropy = entropy
                index = i
                lastE1 = entropy1
                lastE2 = entropy2

        s1["entropy"] = lastE1
        s1["data"] = sortData[: index + 1]
        s2["entropy"] = lastE2
        s2["data"] = sortData[index + 1:]
        return s1, s2, minEntropy

    def train(self, data):
        # 需要遍历的key
        needSplitKey = [0]
        # 将整个数据作为一组
        self.result.setdefault(0, {})
        self.result[0]["entropy"] = np.inf
        self.result[0]["data"] = data
        group = 1
        # 不断更新splitKey
        for key in needSplitKey:
            # print(needSplitKey)
            # 迭代开始为整个特征数据
            S1, S2, entropy = self.split(self.result[key]["data"])
            # 如果满足条件
            if entropy > self.minInfoThreshold and group < self.maxGroup:
                self.result[key] = S1
                newKey = max(self.result.keys()) + 1
                self.result[newKey] = S2
                needSplitKey.extend([key])
                needSplitKey.extend([newKey])
                group += 1
            else:
                break

# if __name__ == "__main__":
#     dbe = DiscreteByEntropy(group=15,threshold=0.5)
#     data = dbe.loadData()
#     dbe.train(data)
#     print("result is {}".format(dbe.result))
#     # print(data)