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

from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV

roc_auc_scorer = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')

def find_best_estimator(classifier,classifier_parameters,cv=5,scoring = roc_auc_scorer,n_jobs=-1):
    grid_search_alg = GridSearchCV(estimator=classifier, param_grid=classifier_parameters, cv=cv, scoring=scoring, refit=True,n_jobs=n_jobs,verbose = 3)
    grid_search_alg.fit(X_train,y_train)
    print('Best Parameters',grid_search_alg.best_params_)
    preds = grid_search_alg.best_estimator_.predict_proba(X_val)
    val_score = roc_auc_score(y_val,preds,multi_class = 'ovr')
    print('Best estimator score',val_score)
    return grid_search_alg.best_estimator_

