📊 **Sales Forecasting Using Machine Learning**
End-to-End ML Pipeline | Feature Engineering | Model Optimization | Stacking | Deployment-Ready
This project predicts Profit based on sales and customer attributes using a complete industry-grade ML workflow.
Built with Python, Scikit-Learn, Pandas, and Joblib.

 **Project Overview**
The goal is to create a Profit Prediction System.
**->The workflow includes:**
Data loading & cleaning
Exploratory Data Analysis (EDA)
Feature engineering
Preprocessing with ColumnTransformer
Model comparison
Hyperparameter tuning (RandomizedSearchCV)
Stacking Regressor
Model evaluation
Deployment using joblib.dump
The final model is saved as:
Copy code
sales_forecasting_model.pkl
You can load and use it anywhere to predict profit.

** Dataset**
Source: Superstore-style retail dataset
-->Target Variable: Profit
-->Main features include:
Sales
Quantity
Discount
Category / Sub-Category
Customer Details
Region / City / State
etc.

 **Tech Stack**
Category	Tools
Language	Python
ML Libraries	Scikit-Learn, Joblib
Data Processing	Pandas, NumPy
Visualization	Matplotlib, Seaborn

**🔍 Key Steps in the Project**
**1️ Data Preprocessing**
Dropped unused columns
Checked missing values
Statistical summary
Heatmap & distribution plots
Numerical & categorical pipelines
StandardScaler + OneHotEncoder

**2️ Model Training**
Tested baseline models:
Linear Regression
KNN Regressor
Evaluated using:
R² Score
RMSE

**3️ Hyperparameter Tuning (RandomizedSearchCV)**
Searched for best KNN parameters:

n_neighbors = [1, 3, 5, 7, 9, 12, 15]
**4️ Stacking Regressor**
Final ensemble:
Linear Regression
Best KNN Model
Final estimator: KNN

**5️ Model Saving**
The entire preprocessing + model pipeline is saved:
lua
Copy code
joblib.dump(pipe3, "sales_forecasting_model.pkl")

**📈 Final Model Performance**
Metrics on test data:
R² Score: Displayed in console after running code
RMSE: Displayed in console after running code
(Values depend on your dataset.)

**How to Use the Saved Model**
python
Copy code
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
📦 Project Structure
css
Copy code
📁 Sales-Forecasting-Project
│── project1.csv
│── sales_forecasting_model.pkl
│── main.py
│── README.md



