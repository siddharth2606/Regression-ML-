from fastapi import FastAPI
import joblib
from pydantic import BaseModel 
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "house_price.pkl")

model = joblib.load(MODEL_PATH)


class houseInput(BaseModel):
    area : float


@app.post("/predict")
def predict_price(data:houseInput):
    input_data = [[data.area]]
    prediction = model.predict(input_data)

    return{
        "prediction": prediction[0]
    }