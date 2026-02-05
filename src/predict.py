import joblib

lr = joblib.load("Regression-ML-/src/house_price.pkl")
prediction = lr.predict([[1360]])
print(prediction)