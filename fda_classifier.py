from bayes_classifier import BayesClassifier
import numpy as np


class FisherLDAClassifier:
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.classes = np.unique(y_train)

    def _calc_means(self):
        global_mean = 0
        num_classes = self.classes.shape[0]
        class_mean = np.zeros((num_classes, self.x_train.shape[1]))
        nk = np.zeros((num_classes, 1))
        n = self.y_train.shape[0]
        for i in range(num_classes):
            class_mean[i, :] = np.mean(self.x_train[np.where(self.y_train == self.classes[i])[0]], 0)
            nk[i] = self.y_train[np.where(self.y_train == self.classes[i])].shape[0]
            global_mean += nk[i] * class_mean[i]
        global_mean /= n
        return global_mean, class_mean, nk

    def _calc_s(self):
        global_mean, class_mean, nk = self._calc_means()
        sw = 0
        sb = 0
        for i in range(self.classes.shape[0]):
            s = np.dot((self.x_train[np.where(self.y_train == self.classes[i])[0]] - class_mean[i]).T,
                       self.x_train[np.where(self.y_train == self.classes[i])[0]] - class_mean[i])
            sw += s
            sb += nk[i] * np.outer((class_mean[i] - global_mean), (class_mean[i] - global_mean).T)
        return sw, sb

    def _calc_eig_vectors(self):
        sw, sb = self._calc_s()
        num_classes = self.classes.shape[0]
        mat = np.dot(np.linalg.pinv(sw), sb)
        eig_value, eig_vector = np.linalg.eig(mat)
        eig_list = [(eig_value[i], eig_vector[:, i]) for i in range(len(eig_value))]
        eig_list = sorted(eig_list, key=lambda x: x[0], reverse=True)
        w = np.array([eig_list[i][1] for i in range(num_classes - 1)])
        return w

    def classify(self):
        w = self._calc_eig_vectors()
        gx = np.dot(self.x_train, w.T)
        gx_t = np.dot(self.x_test, w.T)
        b = BayesClassifier(gx, self.y_train, gx_t)
        return b.classify()
        # num_classes = self.classes.shape[0]
        # class_mean_new = np.zeros((num_classes, gx.shape[1]))
        # for i in range(num_classes):
        #     class_mean_new[i, :] = np.mean(gx[np.where(self.y_train == self.classes[i])[0]], 0)
        # bias = np.array([-(class_mean_new[0, 0] + class_mean_new[2, 0]) / 2,
        #                  -(class_mean_new[0, 1] + class_mean_new[2, 1]) / 2])
        # gx_t = np.dot(self.x_test, w.T) + bias
        # res = np.zeros(self.x_test.shape[0])
        # for i in range(gx_t.shape[0]):
        #     idx = 0
        #     if gx_t[i, 0] > 0 and gx_t[i, 1] < 0:
        #         idx = 0
        #     elif gx_t[i, 0] > 0 and gx_t[i, 1] < 0:
        #         idx = 2
        #     elif gx_t[i, 0] < 0 and gx_t[i, 1] < 0:
        #         idx = 1
        #     elif gx_t[i, 0] > 0 and gx_t[i, 1] > 0:
        #         if gx_t[i, 0] >= gx_t[i, 1]:
        #             idx = 0
        #         else:
        #             idx = 2
        #     res[i] = self.classes[idx]
        # return res
