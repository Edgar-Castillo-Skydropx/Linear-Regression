import numpy as np
from typing import List
from sklearn.linear_model import LinearRegression


def linear_regression(data: List[int], values: List[int]):
    # Variables predictoras
    X = np.array(data)

    # Variable respuesta
    y = np.array(values)

    # Modelo
    model = LinearRegression()
    model.fit(X, y)

    # Coeficientes
    return {
        "model": model,
        "intercept": model.intercept_,
        "coef": model.coef_,
    }
