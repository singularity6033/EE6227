from scipy.stats import norm
import numpy as np


class BayesClassifier:
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.classes = np.unique(y_train)

    def _calc_priori_proba(self):
        priori_idx = self.y_train == self.classes
        priori_proba = np.sum(priori_idx, 0) / self.y_train.shape[0]
        return priori_proba

    def _calc_conditional_priori_dist(self):
        cpd = []
        for i in range(self.classes.shape[0]):
            x_mean = np.mean(self.x_train[np.where(self.y_train == self.classes[i])[0]], 0)
            x_var = np.std(self.x_train[np.where(self.y_train == self.classes[i])[0]], 0)
            cpd.append(norm(x_mean, x_var))
        return cpd

    def classify(self):
        priori_proba = self._calc_priori_proba()
        cpd = self._calc_conditional_priori_dist()
        post_proba = np.zeros((self.x_test.shape[0], self.classes.shape[0]))
        for i in range(self.classes.shape[0]):
            post_proba[:, i] = priori_proba[i] * np.prod(cpd[i].pdf(self.x_test), 1)
        res = self.classes[np.argmax(post_proba, 1)]
        return res

