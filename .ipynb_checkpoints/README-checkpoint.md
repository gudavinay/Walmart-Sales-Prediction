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
