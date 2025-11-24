import pandas as pd
import joblib

#LOAD DATA 
df = pd.read_csv(r"C:\Users\surya\ML_Project\project1.csv")
df.drop(columns=['Order ID','Order Date','Ship Date'], inplace=True)

print(df.head())
print(df.isna().sum())
print(df.describe())
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(df.isna())
plt.show()

sns.histplot(df['Profit'])
plt.show()

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

# FEATURES
X = df.drop(columns=['Profit'])
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num = ['Row ID','Sales','Quantity','Discount']
cat = ['Product Name','Sub-Category','Category','Region','State','City','Country','Segment',
       'Customer Name','Ship Mode','Customer ID','Product ID','Postal Code']

num_line = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scalar', StandardScaler())
])

cat_line = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore'))
])

process = ColumnTransformer([
    ('num', num_line, num),
    ('cat', cat_line, cat)
])

# BASE MODELS TEST
models = {
    'linear': LinearRegression(),
    'knn': KNeighborsRegressor()
}

result = []

for name, m in models.items():
    pipe1 = Pipeline([
        ('process', process),
        ('model', m)
    ])
    pipe1.fit(X_train, y_train)
    pred = pipe1.predict(X_test)
    r2 = r2_score(y_test, pred)
    RMSE = root_mean_squared_error(y_test, pred)
    result.append([name, r2, RMSE])

print(pd.DataFrame(result, columns=['model', 'r2', 'RMSE']))

#RANDOMIZED SEARCH
param_grid = {
    "knn__n_neighbors": [1, 3, 5, 7, 9, 12, 15]
}

pipe2 = Pipeline([
    ('process', process),
    ("knn", KNeighborsRegressor())
])

search = RandomizedSearchCV(
    pipe2,
    param_distributions=param_grid,
    cv=5,
    n_iter=20,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)

print("BEST PARAMS:", search.best_params_)
print("BEST MODEL:", search.best_estimator_)

best_knn_model = search.best_estimator_.named_steps["knn"]

#STACKING
stack = StackingRegressor(
    estimators=[
        ('linear', LinearRegression()),
        ('knn', best_knn_model)
    ],
    final_estimator=KNeighborsRegressor()
)

pipe3 = Pipeline([
    ('process', process),
    ('stack', stack)
])

pipe3.fit(X_train, y_train)

y_pred = pipe3.predict(X_test)

print("Final r2:", r2_score(y_test, y_pred))
print("Final RMSE:", root_mean_squared_error(y_test, y_pred))

# SAVE MODEL
joblib.dump(pipe3, "sales_forecasting_model.pkl")
print("Model saved as: sales_forecasting_model.pkl")
