import pandas as pd
import os

# load the data
housing = pd.read_csv("housing.csv")

# print(housing.head())
# housing.info()
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

import numpy as np
# Divide by 1.5 to limit the number of income categories
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# print(housing["income_cat"])
# Label those above 5 as 5
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# print(housing["income_cat"].value_counts())
# print(housing["income_cat"].hist())

# ======================================================================================
# create a test set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# comparison of stratified versus purely random sampling
from sklearn.model_selection import train_test_split
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
# print(compare_props)

# print(strat_test_set)
# remove the income_cat attribute
for set_ in (strat_train_set, strat_test_set):
    set_.drop(["income_cat"], axis=1, inplace=True)
# drop labels for training set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
# Remove the text attribute
housing_num = housing.drop('ocean_proximity', axis=1)

# build a pipeline for preprocessing the numerical attributes
from sklearn.pipeline import Pipeline
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")), #  Replace missing values with median value
        ('std_scaler', StandardScaler()), # Feature Scaling with StandardScaler
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# build a pipeline for preprocessing the numerical and categorical attributes
from sklearn.compose import ColumnTransformer


from sklearn.preprocessing import OneHotEncoder


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
# ===================================================================================