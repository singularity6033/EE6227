import numpy as np


def MatrixCalculate(x):
    m = np.mean(x, axis=0)
    s = np.zeros((4, 4))
    for i in range(len(x)):
        a = x[i, :] - m
        a = np.array([a])
        b = a.T
        s = s + np.dot(b, a)
    return m, s


class FisherLDAClassifier:
    def __init__(self, x_train, y_train, num_classes=3):
        self.x_train = x_train
        self.y_train = y_train
        self.classes = np.unique(y_train)
        self.n_train = len(self.x_train)
        self.n_class = len(self.classes)

    def classify(self, x_test):
        s = list()
        m = list()
        for c in self.classes:
            x_i = self.x_train[np.where(self.y_train == c)[0]]
            m_i, s_i = MatrixCalculate(x_i)
            s.append(s_i)
            m.append(m_i)
        sw12 = s[0] + s[1]
        sw13 = s[0] + s[2]
        sw23 = s[1] + s[2]
        m1, m2, m3 = m[0], m[1], m[2]
        a = m1 - m2
        a = np.array([a])
        a = a.T
        b = m1 - m3
        b = np.array([b])
        b = b.T
        c = m2 - m3
        c = np.array([c])
        c = c.T
        w12 = (np.dot(np.linalg.inv(sw12), a)).T
        w13 = (np.dot(np.linalg.inv(sw13), b)).T
        w23 = (np.dot(np.linalg.inv(sw23), c)).T
        G12 = -0.5 * (np.dot(np.dot((m1 + m2), np.linalg.inv(sw12)), a))
        G13 = -0.5 * (np.dot(np.dot((m1 + m3), np.linalg.inv(sw13)), b))
        G23 = -0.5 * (np.dot(np.dot((m2 + m3), np.linalg.inv(sw23)), c))
        y_test = []
        for i in range(x_test.shape[0]):
            x = np.array([x_test[i]])
            g12 = np.dot(w12, x.T) + G12
            g13 = np.dot(w13, x.T) + G13
            g23 = np.dot(w23, x.T) + G23
            if g12 > 0 and g13 > 0:
                c = 0
            elif g12 < 0 and g23 > 0:
                c = 1
            elif g13 < 0 and g23 < 0:
                c = 2
            y_test.append(self.classes[c])
        return np.array(y_test)

    def cal_acc(self):
        pred = self.classify(self.x_train)  # test on the training dataset
        pred = np.array(pred)
        truth = self.y_train.reshape(-1, )
        accuracy = np.sum(pred == truth) / truth.shape[0]
        return accuracy
