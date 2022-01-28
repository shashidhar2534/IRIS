import json
import pickle

import numpy as np


__data_columns = None
__model = None


def predict_species(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    x = np.zeros(len(__data_columns))
    x[0] = SepalLengthCm
    x[1] = SepalWidthCm
    x[2] = PetalLengthCm
    x[3] = PetalWidthCm

    return __model.predict([x])[0]


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns


    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']


    global __model
    if __model is None:
        with open('./artifacts/Iris_species.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")




def get_data_columns():
    return __data_columns


if __name__ == '__main__':
    load_saved_artifacts()

    print(predict_species(5.1,3.5,1.4,0.2))