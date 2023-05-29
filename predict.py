import numpy as np
from tensorflow import keras
from typing import List
from create_dataset import fen_to_array

model = keras.models.load_model('model')


def predict(fens: List[str]):
    x = np.array([fen_to_array(i) for i in fens])
    y = model.predict(x)
    return [np.argmax(i) for i in y]
