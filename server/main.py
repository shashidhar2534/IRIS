import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import utils

app = FastAPI()


class Data(BaseModel):
    SepalLengthCm:float
    SepalWidthCm:float
    PetalLengthCm:float
    PetalWidthCm:float

@app.post("/predict")
def predict_iris_species(data: Data):
    response = {
        'Species': utils.predict_species(
            data.SepalLengthCm,
            data.SepalWidthCm,
            data.PetalLengthCm,
            data.PetalWidthCm)}
    return response


if __name__ == "__main__":
    utils.load_saved_artifacts()
    uvicorn.run(app)