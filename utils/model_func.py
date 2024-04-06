import numpy as np


def prediction_price(feature_value, model):
    array = np.full(136, feature_value)
    feature_values = np.array(array).reshape(1, -1)
    return model.predict(feature_values)[0]
