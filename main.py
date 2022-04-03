from scipy.io import loadmat
from bayes_classifier import BayesClassifier
from fda_classifier import FisherLDAClassifier
from dt_classifier import DecisionTreeClassifier


x = loadmat("data/Data_Train.mat")['Data_Train']
y = loadmat("data/Label_Train.mat")['Label_Train']
x_t = loadmat("data/Data_test.mat")['Data_test']
b = BayesClassifier(x, y, x_t)
f = FisherLDAClassifier(x, y, x_t)
d = DecisionTreeClassifier(x, y, x_t)
print('predictions from bayes decision rule: ')
print(b.classify())
print('predictions from Fisher discriminant analysis: ')
print(f.classify())
print('predictions from Decision trees: ')
print(d.classify())