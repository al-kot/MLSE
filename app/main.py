import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model_joblib = joblib.load('./app/regression.joblib')

class ModelInputs(BaseModel):
    size: float
    nb_rooms: int
    garden: int
    
def predict(inputs: ModelInputs):
    return model_joblib.predict([[
        inputs.size,
        inputs.nb_rooms,
        inputs.garden
    ]])[0]

@app.post("/predict")
async def create_prediction(inputs: ModelInputs):
    return predict(inputs)
