import math
import random
import pandas as pd
from collections import defaultdict
from operator import itemgetter

def LoadMovieLensData(filepath, train_rate):
    # 修复：去掉names参数（与header=0冲突，会覆盖原表头）
    ratings = pd.read_csv(filepath, sep=",", header=0, engine='python')
    # 确保列名与CSV实际表头一致（根据实际表头调整）
    ratings = ratings[['userId', 'movieId']]

    train = []
    test = []
    random.seed(3)
    for idx, row in ratings.iterrows():
        try:
            user = int(row['userId'])
            item = int(row['movieId'])
        except ValueError as e:
            print(f"跳过异常行 {idx}：{row}，错误：{e}")
            continue
        if random.random() < train_rate:
            train.append([user, item])
        else:
            test.append([user, item])
    return PreProcessData(train), PreProcessData(test)

def PreProcessData(originData):
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set())
        trainData[user].add(item)
    return trainData


class ItemCF(object):
    """基于物品的协同过滤推荐算法"""
    def __init__(self, trainData, similarity="cosine", norm=True):
        self._trainData = trainData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict()  # 物品相似度矩阵

    def train(self):
        self.similarity()

    def similarity(self):
        N = defaultdict(int)  # 物品被喜爱的用户数
        for user, items in self._trainData.items():
            for i in items:
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items))
        
        # 计算余弦相似度
        for i, related_items in self._itemSimMatrix.items():
            for j, cij in related_items.items():
                self._itemSimMatrix[i][j] = cij / math.sqrt(N[i] * N[j])
        
        # 归一化处理（修复：处理空字典情况）
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                if not relations:  # 跳过空字典，避免报错
                    continue
                max_num = max(relations.values())  # 改用values()更简洁
                if max_num > 0:  # 避免除零
                    self._itemSimMatrix[i] = {k: v/max_num for k, v in relations.items()}

    def recommend(self, user, N, K):
        # 修复：检查用户是否存在
        if user not in self._trainData:
            return f"用户{user}不在训练数据中"
        
        recommends = defaultdict(float)
        items = self._trainData[user]
        
        for item in items:
            # 修复：检查物品是否在相似度矩阵中
            if item not in self._itemSimMatrix:
                continue
            # 取最相似的前K个物品
            for i, sim in sorted(self._itemSimMatrix[item].items(), 
                                key=itemgetter(1), reverse=True)[:K]:
                if i in items:
                    continue
                recommends[i] += sim
        
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

if __name__ == "__main__":
    train, test = LoadMovieLensData("/home/chenyifeng/图片/archive/ratings.csv", 0.8)
    print(f"train data size: {len(train)}, test data size: {len(test)}")
    # 修复：变量名小写，避免与类名冲突
    item_cf = ItemCF(train, similarity='iuf', norm=True)
    item_cf.train()
    
    # 推荐前可先查看训练集中的用户（可选）
    # print("训练集中的用户ID示例：", list(train.keys())[:5])
    print(item_cf.recommend(1, 5, 80))