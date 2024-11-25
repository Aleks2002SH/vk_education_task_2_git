from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.impute import MissingIndicator, SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin

random_state=42

train_df = pd.read_csv('train.csv',index_col = 0)
test_df = pd.read_csv('test.csv',index_col=0)
indicator = MissingIndicator()
train_mask_missing_values_only = indicator.fit_transform(train_df)
test_mask_missing_values_only = indicator.fit_transform(test_df)
print(np.sum(train_mask_missing_values_only))
print(np.sum(test_mask_missing_values_only))
cols = list(train_df.columns)
X = train_df.drop([cols[-1]],axis=1)
y = train_df[cols[-1]]
def find_non_numerical(df):
    non_numerical_cols = df.select_dtypes(exclude=['number']).columns
    if non_numerical_cols.empty:
        print("All columns are numerical.")
    else:
        print(f"Non-numerical columns found: {list(non_numerical_cols)}")
find_non_numerical(train_df)
for col in cols[:-1]:
    number_of_unique_values = len(X[col].unique())
    if number_of_unique_values<X.shape[0]:
        print(f'Columns - {col}, number of unique values - {number_of_unique_values}')
print(f'Unique classes - {y.unique()}')
knn_neighbors = 10
imputer = KNNImputer(n_neighbors = knn_neighbors)
X_imp = imputer.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_imp, y, test_size=0.2, random_state=random_state)

roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

def find_best_estimator(classifier,classifier_parameters,cv=5,scoring = roc_auc_scorer,n_jobs=-1):
    grid_search_alg = GridSearchCV(estimator=classifier, param_grid=classifier_parameters, cv=cv, scoring=scoring, refit=True,n_jobs=n_jobs,verbose = 3)
    grid_search_alg.fit(X_train,y_train)
    print('Best Parameters',grid_search_alg.best_params_)
    preds = grid_search_alg.best_estimator_.predict_proba(X_val)
    val_score = roc_auc_score(y_val,preds,multi_class = 'ovr')
    print('Best estimator score',val_score)
    return grid_search_alg.best_estimator_


class KNN_with_prototypes_classifier(BaseEstimator, ClassifierMixin):
    def __init__(self,random_state = None,n_prototypes = 1,n_neighbors=3,weights='uniform',metric='minkowski'):
        self.prototypes = []
        self.random_state = random_state
        self.n_prototypes = n_prototypes
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
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
        classes_to_classify = np.unique(y_train)
        y_prototypes = []
        for cls in classes_to_classify:
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

best_knn_with_prototypes = KNN_with_prototypes_classifier(random_state=42)
knn_with_prototypes_parameters = {
    'n_neighbors':list(range(3,6)),
    'weights':['distance'],
    'metric':['euclidean','manhattan','chebyshev','cosine'],
    'n_prototypes':[10,20,30]
}
    
best_knn_with_prototypes = find_best_estimator(best_knn_with_prototypes,knn_with_prototypes_parameters)