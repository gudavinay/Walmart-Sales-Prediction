# Team 6 - Walmart Sales Prediction

Members:

- Saketh Reddy Banda (014843881)
- Vinay Guda (015255123)
- Shalaka Bodke (015357069)
- Nikhil Raj Karlapudi (014668121)

Dataset:
Walmart Recruiting Store Sales
https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data?select=stores.csv

Goal:
The goal is to predict weekly department wide sales for different departments in Walmart stores around the US.

Steps:
Data Collection:
Dataset obtained from Kaggle contains Walmart sales data which is categorized into 5 different files namely Features, Stores, Training data, Testing data and Sample Submission. Data that is important for visualization has been considered and joined to create a dataframe using PySpark in the Jupyter notebook. This includes Features, Stores and Training data.

Data PreProcessing:
The dataframe has been checked for duplicate values and found that none of the data has been duplicated. Missing values for each column were calculated, based on the results we observed that there were no missing values present in any of the columns. During the processing of the data, we have observed that some of the sales values were negative and zero in nature, these values were filtered out using spark-SQL. All the pre-processed data was exported to excel using Openpyxl and the same file has been loaded into Tableau. Once the data has been loaded into Tableau, the following Dimensions and Measures was formed.
Code filepath : /dataPreProcessing.ipynb

Data Exploration / Feature Engineering:
The following points were found out in data exploration :
Lot of outliers in Unemployment rate
Importance of IsHoliday field in predicting WeeklySales
Distribution of stores in dataset
Relation between stores and weekly sales
Relation between departments and weekly sales
Weekly Sales distribution over time.
Code filepath : /data_exploration.ipynb & /feature_engineering.ipynb

Models:
KNN Regressor
Code filepath : /KNN.ipynb    
Decision Tree Regressor
Code filepath : /DecisionTree.ipynb
Extra Trees Regressor
Code filepath : /ExtraTreeRegressor.ipynb
Random Forest Regressor
Code filepath : /random-forest.ipynb & randomForest.py
XGBRegressor
Code filepath : /XGB.ipynb

Final Model used - Random Forest Regressor
Code filepath : /random-forest.ipynb & randomForest.py

Frontend development:
Python Flask - We used Python flask which is a micro web development framework in python to create web applications.
Python Pickle - We also used the Python Pickle library to serialize the final model and store it as byte stream in a file using pickle.dump function. Whenever the application made a request prediction of values the model was loaded in the flask app using pickle.load() and the parameters were passed to it.
Code filepath : /app.py & /templates/index.html

Instructions to run:
Set flask path : set flask_app=app.py (Windows)
                export flask_app=app (Mac and Linux)
Run the file from terminal : flask run










