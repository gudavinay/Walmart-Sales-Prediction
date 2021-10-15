import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle


df = pd.read_csv("joined_data_refined.csv",
                 keep_default_na=False, na_values=[""])
print(df.columns)
print(df.info())

del df['MarkDown1']
del df['MarkDown2']
del df['MarkDown3']
del df['MarkDown4']
del df['MarkDown5']
del df['Type_A']
del df['Date']
del df['IsHoliday']
del df['dayofweek_name']
del df['Unnamed: 0']
df.drop('Type_B', axis=1, inplace=True)

# X = df[df.columns.difference(['Weekly_Sales'])]
X = df[['Store', 'Dept']]
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

xg_reg = xgb.XGBRegressor(learning_rate=0.1, max_depth=6,
                          objective='reg:squarederror')
xg_reg.fit(X_train, y_train)

pred = xg_reg.predict(X_train)
y_pred = xg_reg.predict(X_test)

print("xg_reg X_test y_pred = ", type(X_test), X_test.head(), y_pred)

print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print("R-squared score:", r2_score(y_test, y_pred))
print(xg_reg.score(X_test, y_test)*100)

