from scipy.io import loadmat
from bayes_classifier import BayesClassifier
from fda_classifier_new import FisherLDAClassifier
from dt_classifier import DecisionTreeClassifier


x = loadmat("data/Data_Train.mat")['Data_Train']
y = loadmat("data/Label_Train.mat")['Label_Train']
x_t = loadmat("data/Data_test.mat")['Data_test']
b = BayesClassifier(x, y)
f = FisherLDAClassifier(x, y)
d = DecisionTreeClassifier()

print('accuracy of Bayes Classifier on training dataset: ')
print(b.cal_acc())
print('predictions of Bayes Classifier on test dataset: ')
print(b.classify(x_t))

print('accuracy of Fisher Discriminant Analysis on training dataset: ')
print(f.cal_acc())
print('predictions of Fisher Discriminant Analysis on test dataset: ')
print(f.classify(x_t))

d.fit(x, y)
print('accuracy of Decision trees on training dataset: ')
print(d.score(x, y))
print('predictions of Decision trees on test dataset: ')
print(d.predict(x_t))
