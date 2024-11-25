from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.impute import MissingIndicator, SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, make_scorer

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