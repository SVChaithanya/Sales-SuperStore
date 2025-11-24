# Sales Forecasting Using Machine Learning

This project predicts **Profit** using sales and customer details.  
It is a complete end-to-end machine learning workflow including data cleaning, EDA, preprocessing, model training, hyperparameter tuning, stacking, and saving the final model.



## Project Overview
The goal of this project is to build a model that can predict **Profit** based on different features like Sales, Quantity, Discount, Product details, Region, Customer information, etc.

The workflow includes:

- Loading and cleaning the dataset  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Preprocessing using ColumnTransformer  
- Baseline model comparison  
- Hyperparameter tuning using RandomizedSearchCV  
- Stacking Regressor to improve accuracy  
- Saving the final model using joblib  


## Dataset

- Dataset: Superstore-style retail data  
- Rows: 500k+  
- Columns: 14  
- Target variable: **Profit**

Main features:
- Sales  
- Quantity  
- Discount  
- Category / Sub-Category  
- City / State / Region  
- Product and Customer Information  



## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-Learn  
- Matplotlib, Seaborn  
- Joblib  



## Steps in the Project

### 1. Data Preprocessing
- Removed unwanted columns  
- Checked missing values  
- Summary statistics  
- Heatmap and distribution plot  
- Scaling numerical features  
- One-hot encoding categorical features  

### 2. Model Training
Baseline models tested:
- Linear Regression  
- KNN Regressor  

Evaluated using:
- R² Score  
- RMSE  

### 3. Hyperparameter Tuning
RandomizedSearchCV used to tune KNN with values:

```
n_neighbors = [1, 3, 5, 7, 9, 12, 15]
```

### 4. Stacking Regressor
Final model includes:

- Linear Regression  
- Best KNN model  
- Final estimator: KNN  

### 5. Saving the Model

joblib.dump(pipe3, "sales_forecasting_model.pkl")


## Final Model Performance

The performance (R² and RMSE) is printed when the code runs.  
Values depend on your dataset.



## How to Use the Saved Model

import joblib
import pandas as pd

model = joblib.load("sales_forecasting_model.pkl")

sample = pd.DataFrame({
    "Row ID": [1],
    "Sales": [500],
    "Quantity": [2],
    "Discount": [0.1],
    "Product Name": ["Stapler"],
    "Sub-Category": ["Office Supplies"],
    "Category": ["Office Supplies"],
    "Region": ["West"],
    "State": ["California"],
    "City": ["Los Angeles"],
    "Country": ["United States"],
    "Segment": ["Consumer"],
    "Customer Name": ["John Doe"],
    "Ship Mode": ["Second Class"],
    "Customer ID": ["CG-12345"],
    "Product ID": ["OFF-ST-1001"],
    "Postal Code": [90001]
})

prediction = model.predict(sample)
print("Predicted Profit:", prediction[0])


## Project Structure

```
Sales-Forecasting-Project
│── project1.csv
│── main.py
│── sales_forecasting_model.pkl
│── README.md


## Requirements

Create a file named **requirements.txt**:


pandas
numpy
matplotlib
seaborn
scikit-learn
joblib


## Tags
Machine Learning, Regression, Sales Forecasting, RandomizedSearchCV, Stacking Regressor, Data Science Project

