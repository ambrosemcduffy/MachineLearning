import numpy as np
import pandas as pd
import visuals as vs
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Importing in the census data
data = pd.read_csv("data/census.csv")
# Breaking up the data to display income
n_records = len(data["income"])
# number of people who make more than 50K
n_greater_50k = len(data[data["income"] == "<=50K"])
# number of people who make less than 50K
n_at_most_50k = len(data[data["income"] == ">50K"])
# the percentage of people whos income is more than 50k
greater_percent = n_greater_50k/n_records

print("total number of records required: {}".format(n_records))
print("total number of people \who make more than 50K: {}".format(n_greater_50k))
print("total number of people who make less than 50K: {}".format(n_at_most_50k))
print("the percentage of people who make more than 50K: {}".format(greater_percent))

# Splitting the data into features and targets
income_raw = data["income"]
# dropping income out of the dataset
features_raw = data.drop("income", axis=1)
# visualizing out data
#vs.distribution(data)
# breaking up the data we want to skew for logarithm
skewed = ["capital-gain", "capital-loss"]
features_log_transformed = pd.DataFrame(data=features_raw)


def log_trans(x):
    return np.log(x+1)


# apply logarithm function
features_log_transformed[skewed] = log_trans(features_log_transformed[skewed])
#vs.distribution(features_log_transformed, transformed=True)
# Scaling the features
scaler = MinMaxScaler()
numerical = ["age", "education-num", "capital-gain",
             "capital-loss", "hours-per-week"]
features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
# Scaling feats
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
# convert the features to 0 and 1
features_final = pd.get_dummies(features_log_minmax_transform)
income = data["income"].map({"<=50K": 0, ">50K": 1})
encoded = list(features_final.columns)
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    income,
                                                    test_size=0.2,
                                                    random_state=0)
TP = np.sum(income)
FP = income.count()
FN = 0
TN = 0
accuracy = TP/n_records
recall = TP/(TP+FN)
precision = TP/(TP+FP)
fscore = 2*(precision*recall)/(precision+recall)
beta = 0.5
#fbeta = (1+beta)*TP/((1+beta) * TP+beta*FN+FP)
fbeta = ((1+beta) * (precision * recall))/((beta*precision) + recall)
print(fbeta)