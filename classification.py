import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score

# Reading in the spam collection table
# Assigning labels to them
df = pd.read_table("data/SMSSpamCollection.txt",
                   header=None,
                   sep="\t",
                   names=["label", "sms_message"])
# Querying the label Series to map 0, and 1 to names
df["label"] = df.label.map({"ham": 0, "spam": 1})
# pliting up my data into a train and testset
x_train, x_test, y_train, y_test = train_test_split(df["sms_message"],
                                                    df["label"],
                                                    random_state=1)
# Tokenizing my texts
count_vector = CountVectorizer()
# fitting the train and testsets
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

naive_bayes = MultinomialNB()
bag_mod = BaggingClassifier(n_estimators=200)
rf_mod = RandomForestClassifier(n_estimators=200)
ada_mod = AdaBoostClassifier(n_estimators=300, learning_rate=0.2)
svm_mod = SVC()

naive_bayes.fit(training_data, y_train)
bag_mod.fit(training_data, y_train)
rf_mod.fit(training_data, y_train)
ada_mod.fit(training_data, y_train)
svm_mod.fit(training_data, y_train)

nv_pred = naive_bayes.predict(testing_data)
bg_pred = bag_mod.predict(testing_data)
rf_pred = rf_mod.predict(testing_data)
ad_pred = ada_mod.predict(testing_data)
svm_pred = svm_mod.predict(testing_data)


def accuracy(preds, actual):
    # finding out how the prediction matches y_test
    # finding how many we got wrong
    return np.sum(preds == actual)/len(actual)


def precision(preds, actual):
    tp = len(np.intersect1d(np.where(actual == 1),
                            np.where(preds == 1)))
    preds_pos = np.sum(preds == 1)
    return tp/preds_pos


def recall(preds, actual):
    tp = len(np.intersect1d(np.where(actual == 1),
                            np.where(preds == 1)))
    ap = np.sum(actual == 1)
    return tp/ap


def f1(preds, actual):
    pre = precision(preds, actual)
    rec = recall(preds, actual)
    return 2*(pre*rec)/(pre+rec)


print(f1(nv_pred, y_test))
print(fbeta_score(y_test, nv_pred, beta=1))
