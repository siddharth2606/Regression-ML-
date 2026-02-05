import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_loader import load_data
import joblib

dataset = load_data("Regression-ML-/dataset/house_price_dataset.csv")
x = dataset[["area_sqft"]]
y = dataset["price"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(x_train,y_train)

joblib.dump(model,"house_price.pkl")