from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.impute import MissingIndicator, SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer

random_state=42

train_df = pd.read_csv('train.csv',index_col = 0)
cols = list(train_df.columns)
X = train_df.drop([cols[-1]],axis=1)
y = train_df[cols[-1]]

knn_neighbors = 10
imputer = KNNImputer(n_neighbors = knn_neighbors)
X_imp = imputer.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_imp, y, test_size=0.2, random_state=random_state)

from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin

class KNN_with_prototypes_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,random_state = None,n_prototypes = 1,n_neighbors=3,weights='uniform',metric='minkowski'):
        self.prototypes = []
        self.random_state = random_state
        self.n_prototypes = n_prototypes
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.classes_ = None
        self.knn = KNeighborsClassifier(n_neighbors = n_neighbors,weights = weights,metric = metric)
        pass
    def get_params(self, deep=True):
        return {
            'random_state': self.random_state,
            'n_prototypes': self.n_prototypes,
            'n_neighbors': self.n_neighbors,
            'weights': self.weights,
            'metric': self.metric
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        if 'n_neighbors' in params or 'weights' in params or 'metric' in params:
            self.knn = KNeighborsClassifier(
                n_neighbors=self.n_neighbors,
                weights=self.weights,
                metric=self.metric
            )
        return self
    def fit(self, x_train, y_train):
        self.prototypes = []
        self.classes_ = np.unique(y_train)
        y_prototypes = []
        for cls in self.classes_:
            x_train_cls = x_train[y_train==cls]
            kmeans = KMeans(n_clusters=self.n_prototypes, random_state=self.random_state)
            kmeans.fit(x_train_cls)
            self.prototypes.append(kmeans.cluster_centers_)
            y_prototypes.append(np.full(self.n_prototypes, cls))
        y_prototypes = np.hstack(y_prototypes)
        self.prototypes = np.vstack(self.prototypes)
        self.knn.fit(self.prototypes,y_prototypes)
        return self
    def return_etalons(self):
        return self.prototypes
    def predict(self, x_test):
        return self.knn.predict(x_test)
    def predict_proba(self, x_test):
        return self.knn.predict_proba(x_test)