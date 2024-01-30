import pandas as pd
import numpy as np

def create_dataset():
    np.random.seed(0)
    data = {
        'X1': np.random.rand(100),
        'X2': np.random.rand(100),
        'Y': np.random.rand(100)
    }
    df = pd.DataFrame(data)
    return df
