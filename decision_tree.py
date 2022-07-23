import numpy as np
from copy import deepcopy

class DecisionTree:

    def __init__(self,
                 Selected_method,
                 max_depth,
                 min_samples_leaves,
                 SampleFeature=False):
        '''
        Input:
            Selected_method: 计算不纯度的方法：信息增益，信息增益率，基尼系数
            max_depth: 所构建决策树的最大深度
            min_samples_leaves: 允许决策树某一结点存在分支的最小样本个数
            SampleFeature:决定是否需要对样本的特征进行抽样
        '''
        if Selected_method == 'infogain_ratio':
            self.Selected_method = self.cal_information_gain_ratio
        elif Selected_method == 'entropy':
            self.Selected_method = self.cal_information_gain
        elif Selected_method == 'gini':
            self.Selected_method = self.cal_gini_purification
        self.tree = None
        self.max_depth = max_depth
        self.min_samples_leaves = min_samples_leaves
        self.SampleFeature = SampleFeature

    def fit(self, X, y, sample_weights=None):
        """
        Input:
            X: 训练集中各样本的特征元素组成的集合
            y: 训练集中各样本的类别标签
            sample_weights: 各样本的权重
        """
        if sample_weights is None:
            # 默认的权重应全为0
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            # 若已给出权重，应对其归一化
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self.tree = self.CreateTree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        # 计算熵值
        entropy = 0.0

        if y.size <= 1:
            return float(0)

        labels = np.unique(y)               # numpy中unique函数用于只保留不同的值，这里用以计算熵值
        N = labels.shape[0]
        weight_sums = np.zeros((N,))

        for n in range(N):
            indices = np.argwhere(
                np.array(y == labels[n]) == True)
            weight_sums[n] = np.sum(sample_weights[indices])

        for weight_sum in weight_sums:
            entropy += np.multiply(weight_sum, np.log(weight_sum))

        entropy = -entropy

        return entropy

    def cal_information_gain(self, X, y, index, sample_weights):
        # 计算信息增益，index是所计算的信息增益特征的索引值
        info_gain = 0

        X_ent = self.entropy(y, sample_weights)
        feature_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        sub_ent = 0

        for feature_val, cnt in zip(feature_vals, val_cnt):
            sample_indices = np.argwhere(np.array(X[:, index] == feature_val) == True)
            divided_sample_weights = sample_weights[sample_indices]
            divided_weights_sum = np.sum(divided_sample_weights)
            divided_sample_weights /= divided_weights_sum
            sub_ent += divided_weights_sum * self.entropy(y[sample_indices], divided_sample_weights)

        info_gain = X_ent - sub_ent

        return info_gain
    
    def cal_intrinsic_value(self, X, y, index, sample_weights):
        # 计算做出分类之前的值

        feature_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        initial_val = 0

        for feature_val, cnt in zip(feature_vals, val_cnt):
            sample_indices = np.argwhere(
                np.array(X[:, index] == feature_val) == True)
            divided_sample_weights = sample_weights[sample_indices]
            divided_weights_sum = np.sum(divided_sample_weights)
            initial_val += divided_weights_sum * np.log(divided_weights_sum)
        
        initial_val = -initial_val

        return initial_val

    def cal_information_gain_ratio(self, X, y, index, sample_weights):
        # 计算信息增益率(与信息增益相比需要除以做出分类前的熵值)
        info_gain_ratio = 0

        info_gain = self.cal_information_gain(X, y, index, sample_weights)
        intrinsic_value = self.cal_intrinsic_value(X, y, index, sample_weights)

        if intrinsic_value != 0:
            info_gain_ratio = info_gain / intrinsic_value
        else:
            info_gain_ratio = np.Infinity

        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights):
        # 计算基尼系数用以刻画数据的纯净程度

        if y.size <= 1:
            return float(0)

        gini = 0

        labels = np.unique(y)

        N = labels.shape[0]
        weight_sums = np.zeros((N,))

        for n in range(N):
            indices = np.argwhere(
                np.array(y == labels[n]) == True)
            weight_sums[n] = np.sum(sample_weights[indices])
        
        for weight_sum in weight_sums:
            gini += np.square(weight_sum)

        gini = - gini
        
        gini += 1

        return gini

    def cal_gini_purification(self, X, y, index, sample_weights):
        # 计算分类之后的基尼系数，以此判断何种特征最好
        new_impurity = 1

        feature_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        new_impurity = 0

        for feature_val, cnt in zip(feature_vals, val_cnt):
            sample_indices = np.argwhere(
                np.array(X[:, index] == feature_val) == True)
            divided_sample_weights = sample_weights[sample_indices]
            divided_weights_sum = np.sum(divided_sample_weights)
            divided_sample_weights /= divided_weights_sum
            new_impurity += divided_weights_sum * \
                self.gini_impurity(y[sample_indices], divided_sample_weights)

        return new_impurity

    def SplitData(self, X, y, index, value, sample_weights):
        # 根据索引与值按其对应特征划分原数据集
        divided_X, divided_y, divided_sample_weights = X, y, sample_weights

        sample_indices = np.argwhere(np.array(X[:, index] == value) == True)
        sample_indices = sample_indices.ravel()
        divided_X = X[sample_indices]
        divided_y = y[sample_indices]
        divided_sample_weights = sample_weights[sample_indices]
        
        divided_X = np.delete(divided_X, index, axis=1)
        divided_sample_weights /= np.sum(divided_sample_weights)

        return divided_X, divided_y, divided_sample_weights

    def ChooseBestFeature(self, X, y, sample_weights):
        # 根据所选的评判标准：信息增益，信息增益率，基尼系数，来选取对应的最优划分特征

        best_feature_index = 0
        D = X.shape[1]

        if D <= 1:
            return 0

        if self.SampleFeature == False:
            if self.Selected_method == self.cal_gini_purification:
                score = np.ones((D,))

                for d in range(D):
                    score[d] = self.Selected_method(X, y, d, sample_weights)

                best_feature_index = np.argmin(score)
            else:
                score = np.zeros((D,))

                for d in range(D):
                    score[d] = self.Selected_method(X, y, d, sample_weights)

                best_feature_index = np.argmax(score)
        else:
            sample_feature_size = np.rint(np.sqrt(D)).astype(int)
            sample_feature_indices = np.random.choice(D, size=sample_feature_size, replace=False)
            sample_feature_indices = np.sort(sample_feature_indices)

            if self.Selected_method == self.cal_gini_purification:
                score = np.ones((D,))

                for i in range(sample_feature_size):
                    score[sample_feature_indices[i]] = self.Selected_method(
                        X, y, sample_feature_indices[i], sample_weights)

                best_feature_index = np.argmin(score)
            else:
                score = np.zeros((D,))

                for i in range(sample_feature_size):
                    score[sample_feature_indices[i]] = self.Selected_method(X, y, sample_feature_indices[i], sample_weights)

                best_feature_index = np.argmax(score)

        return best_feature_index

    @staticmethod
    def majority_vote(y, sample_weights=None):
        # 当结点不满足继续往下划分的条件时，按照少数服从多数为该结点划分标签
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        majority_label = y[0]

        y_signed = deepcopy(y)
        y_signed = np.where(y_signed == 0, -1, 1)

        # 带权重的标签
        y_signed = sample_weights * y_signed

        majority_label = np.sign(np.sum(y_signed))

        if majority_label > 0:
            majority_label = 1
        else:
            majority_label = 0

        return majority_label

    def CreateTree(self, X, y, feature_names, depth, sample_weights):
        '''
        建树流程：（整个过程按照字典的键值对进行建树）
        1、找到最优的划分特征，将此特征作为字典名，在其键值上递归建树
        2、根据所选的最优特征对原数据集进行划分
        3、针对特殊情况讨论：
            a.若没有其他特征可选（所有特征已被遍历），根据少数服从多数为结点分配标签
            b.若达到了所设定的最大深度或结点下的样本数未达到继续进行分支的条件，仍按照上述准则进行处理
        4、递归建树
        '''
        feature_names = deepcopy(feature_names)

        myTree = dict()
        best_feature_index = self.ChooseBestFeature(X, y, sample_weights)
        best_feature_name = feature_names[best_feature_index]
        myTree[best_feature_name] = {}
        feature_names.remove(best_feature_name)

        feature_dict = dict()
        best_feature_col = X[:, best_feature_index]
        best_feature_vals = np.unique(best_feature_col)

        for best_feature_val in best_feature_vals:
            feature_dict[best_feature_val] = {}
            X_divided, y_divided, sample_weights_sub = self.SplitData(X, y, best_feature_index, best_feature_val, sample_weights)
            if X_divided.shape[1] == 0:
                # 若所有特征已被遍历
                feature_dict[best_feature_val] = self.majority_vote(y_divided, sample_weights=sample_weights_sub)
            elif X_divided.shape[0] < self.min_samples_leaves:
                # 若结点下的样本数未达到继续进行分支的标准
                feature_dict[best_feature_val] = self.majority_vote(y_divided, sample_weights=sample_weights_sub)
            elif (depth == self.max_depth) or (y_divided == y_divided[0]).all(): # or (X_sub == X_sub[0]).all():
                # 若达到了最大深度
                feature_dict[best_feature_val] = self.majority_vote(y_divided, sample_weights=sample_weights_sub)
            else:
                # 一般情况：递归建树
                feature_dict[best_feature_val] = self.CreateTree(X_divided, y_divided, feature_names, depth+1, sample_weights_sub)
            
        myTree[best_feature_name] = feature_dict

        return myTree

    def predict(self, X):
        # 根据特征预测分类

        def Classify(tree, x):
            # 对单个特征在已构建的决策树上进行划分
            predict_label = 0

            test_feature = (list(tree.keys()))[0]
            test_feature_dict = tree.get(test_feature)
            test_feature_vals = list(test_feature_dict.keys())

            test_feature_index = feature_names.index(test_feature)
            x_test_feature_val = x[test_feature_index]

            if x_test_feature_val in test_feature_vals:
                pass
            else:
                import random 
                x_test_feature_val = random.choice(test_feature_vals)

            Childtree = test_feature_dict.get(x_test_feature_val)
            if type(Childtree) is dict:
                predict_label = Classify(deepcopy(Childtree), x)
            else:
                predict_label = Childtree

            return predict_label

    
        N, D = X.shape
        predictions = np.zeros((N,))
        feature_names = X.columns.tolist()

        X = X.values

        nextNode = deepcopy(self.tree)

        for i in range(N):
            predictions[i] = Classify(nextNode, X[i])
        
        return predictions

    def show(self):
        import tree_plotter
        tree_plotter.createPlot(self.tree)