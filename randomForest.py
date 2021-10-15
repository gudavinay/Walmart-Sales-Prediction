from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


df =  pd.read_csv("joined_data_refined.csv",keep_default_na=False, na_values=[""])
print(df.columns)

X = df.loc[:, df.columns != 'Weekly_Sales']
y = df.loc[:, df.columns == 'Weekly_Sales']

X = X[["Store", "Dept", "Size", "IsHoliday", "CPI", "Temperature","Type_B","Type_C","MarkDown4","month","Year" ]]
y = y.values.reshape(-1, 1)
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=58, max_depth=27, min_samples_split=3, min_samples_leaf=1)
rf.fit(X_train, y_train.ravel())
print('Accuracy:',rf.score(X_test, y_test.ravel())*100,'%')

y_pred = rf.predict(X_test)

rms = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:',rms)

print('MAE:',mean_absolute_error(y_test, y_pred))

filename = 'finalized_model.sav'
pickle.dump(rf, open(filename, 'wb'))
