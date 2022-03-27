from sklearn import tree
from sklearn.tree import export_text


class DecisionTreeClassifier:
    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test

    def classify(self):
        clf = tree.DecisionTreeClassifier()
        clf.fit(self.x_train, self.y_train)
        print(export_text(clf))
        prediction_results = clf.predict(self.x_test)
        return prediction_results
