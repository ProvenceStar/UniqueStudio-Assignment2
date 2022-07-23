import copy
import numpy as np
from decision_tree import DecisionTree


class Adaboost:

    def __init__(self,
                 BasicModel,
                 estimator_num,
                 seed=10086):
        '''
        BasicModel: 构成adaboost的基本模型
        estimator_num: adaboost模型中所包含的基本模型个数
        '''
        np.random.seed(seed)
        self.BasicModel = BasicModel
        self.estimator_num = estimator_num
        self.estimators = [copy.deepcopy(self.BasicModel) for _ in range(self.estimator_num)]
        self.alphas = [1 for _ in range(estimator_num)]

    def fit(self, X, y):
        """
        通过所给数据训练模型
        """

        N, D = X.shape

        # 初始权重为 1/N
        weights = np.full((N,), np.divide(1, N))

        index = 0

        for estimator in self.estimators:
            # 先创建一个弱分类机
            estimator.fit(X, y, sample_weights=weights)
            # 根据弱分类机进行预测
            y_pred = estimator.predict(X)
            # 根据预测结果得到错误率
            loss_weight = np.sum(weights * np.where(y == y_pred, 0, 1))
            ErrorRate = np.divide(loss_weight, np.sum(weights))
            # 更新alpha值
            self.alphas[index] = np.log(np.divide(1-ErrorRate, ErrorRate))
            # 更新权重（对错误的分类加大其权重）
            weights = weights * np.exp(self.alphas[index] * np.where(y == y_pred, 0, 1))
            
            index += 1

        return self

    def predict(self, X):
        """
        利用已训练好的模型进行分类预测
        """
        N = X.shape[0]
        y_predict = np.zeros(N)

        N, _ = X.shape
        predictions = np.zeros((self.estimator_num, N))

        index = 0

        for estimator in self.estimators:
            predictions[index] = estimator.predict(X)
            index += 1

        for i in range(N):
            y_predict[i] = DecisionTree.majority_vote(
                predictions[:, i].reshape(self.estimator_num,), 
                sample_weights=np.array(self.alphas))

        return y_predict
