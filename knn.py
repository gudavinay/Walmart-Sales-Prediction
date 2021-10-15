from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
df = pd.read_csv("joined_data_refined.csv",
                 keep_default_na=False, na_values=[""])
df.isnull().sum()
sns.set_theme()

df['year'] = pd.to_datetime(df['Date'], format="%Y-%m-%d").dt.year.astype(int)
df['day'] = pd.to_datetime(df['Date'], format="%Y-%m-%d").dt.day.astype(int)
del df['Unnamed: 0']
del df['Date']
del df['is_weekend']
del df['Type_A']
del df['quarter']
del df['MarkDown4']
del df['dayofweek_name']
print(df.head())
corr = df.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True)
X = df.copy()
del X['Weekly_Sales']
y = df['Weekly_Sales']
y = y.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
X_train.columns
X_train.head()
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
knn = KNeighborsRegressor(n_neighbors=10, n_jobs=4)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
plt.scatter(y_test, y_pred_knn)
plt.show()
print(mean_absolute_error(y_test, y_pred_knn))
print(mean_squared_error(y_test, y_pred_knn))
print(knn.score(X_test, y_test)*100)
