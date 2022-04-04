import numpy as np
from sklearn import tree
from sklearn.tree import export_text
import numpy as np
import math
import pandas as pd

# from sklearn API
# class DecisionTreeClassifier:
#     def __init__(self, x_train, y_train):
#         self.x_train = x_train
#         self.y_train = y_train
#
#     def classify(self, x_test):
#         clf = tree.DecisionTreeClassifier()
#         clf.fit(self.x_train, self.y_train)
#         # print(export_text(clf))
#         prediction_results = clf.predict(x_test)
#         return prediction_results
#
#     def cal_acc(self):
#         pred = self.classify(self.x_train)  # test on the training dataset
#         pred = np.array(pred)
#         truth = self.y_train.reshape(-1, )
#         accuracy = np.sum(pred == truth) / truth.shape[0]
#         return accuracy


class TreeNode:
    def __init__(self, data, output):
        self.data = data
        self.children = {}
        self.output = output
        self.index = -1

    def add_child(self, feature_value, obj):
        self.children[feature_value] = obj


class DecisionTreeClassifier:
    def __init__(self):
        # root represents the root node of the decision tree built after fitting the training data
        self.__root = None

    @staticmethod
    def __count_unique(Y):
        # returns a dictionary with keys as unique values of Y(i.e no of classes) and the corresponding value as its
        # frequency
        d = {}
        for i in Y:
            if i not in d:
                d[i] = 1
            else:
                d[i] += 1
        return d

    def __entropy(self, Y):
        # returns the entropy
        freq_map = self.__count_unique(Y)
        entropy_ = 0
        total = len(Y)
        for i in freq_map:
            p = freq_map[i] / total
            entropy_ += (-p) * math.log2(p)
        return entropy_

    def __gain_ratio(self, X, Y, selected_feature):
        # returns the gain ratio
        info_orig = self.__entropy(Y)
        info_f = 0
        split_info = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        initial_size = df.shape[0]
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            info_f += (current_size / initial_size) * self.__entropy(df1[df1.shape[1] - 1])
            split_info += (-current_size / initial_size) * math.log2(current_size / initial_size)

        # to handle the case when split info = 0 which leads to division by 0 error
        if split_info == 0:
            return math.inf

        info_gain = info_orig - info_f
        gain_ratio = info_gain / split_info
        return gain_ratio

    def __gini_index(self, Y):
        # returns the gini index
        freq_map = self.__count_unique(Y)
        gini_index_ = 1
        total = len(Y)
        for i in freq_map:
            p = freq_map[i] / total
            gini_index_ -= p ** 2
        return gini_index_

    def __gini_gain(self, X, Y, selected_feature):
        # returns the gini gain
        gini_orig = self.__gini_index(Y)
        gini_split_f = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        initial_size = df.shape[0]
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            gini_split_f += (current_size / initial_size) * self.__gini_index(df1[df1.shape[1] - 1])

        gini_gain_ = gini_orig - gini_split_f
        return gini_gain_

    def __decision_tree(self, X, Y, features, level, metric, classes):
        # returns the root of the Decision Tree(which consists of TreeNodes) built after fitting the training data
        if len(set(Y)) == 1:
            output = None
            for i in classes:
                if i in Y:
                    output = i

            return TreeNode(None, output)

        if len(features) == 0:
            freq_map = self.__count_unique(Y)
            output = None
            max_count = -math.inf
            for i in classes:
                if i in freq_map:
                    if freq_map[i] > max_count:
                        output = i

            return TreeNode(None, output)

        max_gain = -math.inf
        final_feature = None
        for f in features:
            if metric == "gain_ratio":
                current_gain = self.__gain_ratio(X, Y, f)
            elif metric == "gini_index":
                current_gain = self.__gini_gain(X, Y, f)

            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = f

        freq_map = self.__count_unique(Y)
        output = None
        max_count = -math.inf

        for i in classes:
            if i in freq_map:
                if freq_map[i] > max_count:
                    output = i
                    max_count = freq_map[i]

        unique_values = set(X[:, final_feature])
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y

        current_node = TreeNode(final_feature, output)
        index = features.index(final_feature)
        features.remove(final_feature)
        for i in unique_values:
            df1 = df[df[final_feature] == i]
            node = self.__decision_tree(df1.iloc[:, 0:df1.shape[1] - 1].values, df1.iloc[:, df1.shape[1] - 1].values,
                                        features, level + 1, metric, classes)
            current_node.add_child(i, node)

        features.insert(index, final_feature)

        return current_node

    def fit(self, X, Y, metric="gain_ratio"):
        # Fits to the given training data
        features = [i for i in range(len(X[0]))]
        Y = Y.reshape(-1)
        classes = set(Y)
        level = 0
        if metric != "gain_ratio":
            if metric != "gini_index":
                metric = "gain_ratio"
        self.__root = self.__decision_tree(X, Y, features, level, metric, classes)

    def __predict_for(self, data, node):
        if len(node.children) == 0:
            return node.output

        val = data[node.data]
        if val not in node.children:
            return node.output

        return self.__predict_for(data, node.children[val])

    def predict(self, X):
        # This function returns Y predicted
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.__predict_for(X[i], self.__root)
        return np.array(Y)

    def score(self, X, Y):
        # returns the mean accuracy
        Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                count += 1
        return count / len(Y_pred)
