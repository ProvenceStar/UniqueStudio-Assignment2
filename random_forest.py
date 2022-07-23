import copy
import numpy as np
from decision_tree import DecisionTree


class RandomForest:

    def __init__(self,
                 BasicModel,
                 estimator_num,
                 seed=10086):
        '''
        Input:
        BasicModel: 随机森林中的单个决策树模型
        estimator_num: 随机森林中所包含的决策树的个数
        '''
        np.random.seed(seed)
        self.BasicModel = BasicModel
        self.estimator_num = estimator_num
        self.estimators = [copy.deepcopy(self.BasicModel) for _ in range(self.estimator_num)]

    def GetBootStrapData(self, X, y):
        """
        Bootstrap——自助法:
        一种有放回的抽样方法，具体操作如下：
        （1） 采用重抽样技术从原始样本中抽取一定数量（自己给定）的样本，此过程允许重复抽样
        （2） 根据抽出的样本计算给定的统计量T
        （3） 重复上述N次（一般大于1000），得到N个统计量T
        （4） 计算上述N个统计量T的样本方差，得到统计量的方差
        """

        N, D = X.shape
        X_BootStrap = np.zeros((N, D))
        y_BootStrap = np.zeros((N,))

        choice_indices = np.random.choice(N, size=N, replace=True)
        choice_indices = np.sort(choice_indices)

        X_BootStrap = X.iloc[choice_indices]
        y_BootStrap = y.iloc[choice_indices]

        return X_BootStrap, y_BootStrap

    def fit(self, X, y):
        # 根据BootStrap所得数据进行建树

        # 创建随机森林
        for estimator in self.estimators:
            X_train, y_train = self.GetBootStrapData(X, y)
            estimator.fit(X_train, y_train)

        return self

    def predict(self, X):
        # 根据上述所得的各决策树模型构成随机森林进行预测
        
        N = X.shape[0]
        y_predict = np.zeros(N)

        N, _ = X.shape
        predictions = np.zeros((self.estimator_num, N))

        index = 0

        for estimator in self.estimators:
            predictions[index] = estimator.predict(X)
            index += 1
        
        for i in range(N):
            y_predict[i] = DecisionTree.majority_vote(predictions[:, i].reshape(self.estimator_num,))

        return y_predict