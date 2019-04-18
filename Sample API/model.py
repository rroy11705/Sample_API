import sklearn.model_selection 
from sklearn.neighbors import KNeighborsClassifier
import pickle

class KNeighbor_Model(object):
    
    def __init__(self):
        self.clf = KNeighborsClassifier(n_neighbors = 3)

    def train(self, X, Y):
        self.clf.fit(X, Y)
    
    def score(self, X, Y):
        score1 = self.clf.score(X, Y)
        return score1

    def predict(self, Y):
        survived = self.clf.predict(Y)
        return survived
    
    def predict_proba(self, Y):
        acc = self.clf.predict_proba(Y)
        return acc
    
    def pickle_clf(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)
            print("pickle classifer at {}".format(path))
    
