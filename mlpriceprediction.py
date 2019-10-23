''' 
This is the Machine Learning Project of Home Price Prediction 
it contain 13 labels and 1 feature that decides the predicted output
For users go to the end of the programme and make changes in Using the model 
labels and get the predicted price by own .I attached housing.names and
housing.data in files and by analysis i found output of the different models
best is RandomForestRegressor model.

I attached data.csv file that is excel file you can edit the data of that file
and test with the different input data's and get the different output.


'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load


housing = pd.read_csv("data.csv")
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)

#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:] 
#     return data.iloc[train_indices], data.iloc[test_indices]
train_set, test_set  = train_test_split(housing, test_size=0.2, random_state=42)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

a = strat_test_set['CHAS'].value_counts()
b = strat_train_set['CHAS'].value_counts()
# print(a)
# print(b)
housing = strat_train_set.copy()
# Looking for correlations
corr_matrix = housing.corr()
a=corr_matrix['MEDV'].sort_values(ascending=False)
# print(a)

# Trying out attribute combinations
housing["TAXRM"] = housing['TAX']/housing['RM']
# print(housing.head())
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Missing Attributes
housing.dropna(subset=["RM"])
# print(a.shape)
# print(housing.drop("RM", axis=1).shape)
# a=housing.head()
# print(a)
# a =housing['CHAS'].value_counts()
# print(a)
median = housing["RM"].median()
housing["RM"].fillna(median)


imputer = SimpleImputer(strategy="median")
imputer.fit(housing)


X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns=housing.columns)

# print(housing_tr.describe())

# Creating a Pipeline
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)
# print(housing_num_tr.shape)

# Selecting a desired model for The Home Price Predictor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor() #This is the best model for this type of problem it gives minimum rms values that can increase it's performance of predictions.
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)

model.predict(prepared_data)
list(some_labels)

# Evaluting the model
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
# print(rmse)

# Using better evalution technique-Cross Validation
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
# print(rmse_scores)
def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())
# print(print_scores(rmse_scores))

# Saving The model
dump(model, 'Dragon.joblib') 

# Testing the model on test data

X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))
# print(final_rmse)
prepared_data[0]
# print(prepared_data[0])

# Using the model

model = load('Dragon.joblib') 
features = np.array([[-5.43942006, 4.12628155, -1.6165014, -0.67288841, -1.42262747,
       -11.44443979304, -4.31238772,  7.61111401, -26.0016879 , -0.5778192 ,
       -0.97491834,  0.41164221, -6.86091034]])
print(model.predict(features))



