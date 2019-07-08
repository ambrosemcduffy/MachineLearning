from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Importing in the data
boston = load_boston()
y = boston.target
x = boston.data
# Splitting the data into train and testsets
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.33,
                                                    random_state=42)
# Instantiating the models with default values
linear_model = LinearRegression()
rand_forest = RandomForestRegressor()
adaboost_model = AdaBoostRegressor()
decision_model = DecisionTreeRegressor()

linear_model.fit(x_train, y_train)
rand_forest.fit(x_train, y_train)
adaboost_model.fit(x_train, y_train)
decision_model.fit(x_train, y_train)

preds_rf = rand_forest.predict(x_test)
preds_linear = linear_model.predict(x_test)
preds_ada = adaboost_model.predict(x_test)
preds_dt = decision_model.predict(x_test)

print(r2_score(y_test, preds_rf))
print(mean_squared_error(y_test, preds_rf))
print(mean_absolute_error(y_test, preds_rf))
