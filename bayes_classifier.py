from scipy.stats import norm
import numpy as np


class BayesClassifier:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = np.unique(y_train)

    def _calc_priori_proba(self):
        priori_idx = self.y_train == self.classes
        priori_proba = np.sum(priori_idx, 0) / self.y_train.shape[0]
        return priori_proba

    def _calc_conditional_priori_dist_param(self):
        # multivariate normal density function
        cpd = []
        for i in range(self.classes.shape[0]):
            x_data = self.x_train[np.where(self.y_train == self.classes[i])[0]]
            x_mean = np.mean(x_data, 0)
            x_cov = np.cov(x_data.T)
            cpd.append((x_mean, x_cov))
        return cpd

    def classify(self, x_test):
        priori_proba = self._calc_priori_proba()
        cpd = self._calc_conditional_priori_dist_param()
        c_n = self.classes.shape[0]
        x_n = x_test.shape[0]
        post_proba = np.zeros((x_n, c_n))
        for xt in range(x_n):
            for i in range(c_n):
                xm = cpd[i][0]
                xc = cpd[i][1]
                post_proba[xt, i] = priori_proba[i] * (1 / (((2 * np.pi) ** 2) * np.linalg.det(xc) ** 0.5)) \
                                    * np.exp(np.dot(np.dot(-0.5 * (x_test[xt, :] - xm).T, np.linalg.inv(xc)),
                                                    (x_test[xt, :] - xm)))
        res = self.classes[np.argmax(post_proba, 1)]
        return res

    def cal_acc(self):
        pred = self.classify(self.x_train)  # test on the training dataset
        pred = np.array(pred)
        truth = self.y_train.reshape(-1, )
        accuracy = np.sum(pred == truth) / truth.shape[0]
        return accuracy
