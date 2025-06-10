import numpy as np
from typing import List, Dict, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression(
    X: List[List[float]], y: List[float]
) -> Dict[str, Union[LinearRegression, float, np.ndarray]]:
    """
    Entrena un modelo de regresión lineal.

    Args:
        X: Variables predictoras (matriz de características)
        y: Variable objetivo

    Returns:
        dict: Diccionario con el modelo entrenado y sus métricas

    Raises:
        ValueError: Si los datos están vacíos o tienen dimensiones inconsistentes
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Los datos no pueden estar vacíos")

    if len(X) != len(y):
        raise ValueError("X e y deben tener la misma cantidad de muestras")

    # Convertir a arrays de numpy
    X_array = np.array(X)
    y_array = np.array(y)

    # Asegurar que X sea 2D
    if X_array.ndim == 1:
        X_array = X_array.reshape(-1, 1)

    # Crear y entrenar el modelo
    model = LinearRegression()
    model.fit(X_array, y_array)

    # Calcular métricas
    y_pred = model.predict(X_array)
    mse = mean_squared_error(y_array, y_pred)
    r2 = r2_score(y_array, y_pred)

    return {
        "model": model,
        "intercept": model.intercept_,
        "coef": model.coef_,
        "r2_score": r2,
        "mse": mse,
        "rmse": np.sqrt(mse),
    }


def predict_value(model_dict: Dict, features: List[float]) -> float:
    """
    Realiza una predicción usando el modelo entrenado.

    Args:
        model_dict: Diccionario retornado por linear_regression()
        features: Lista de características para predecir

    Returns:
        float: Valor predicho
    """
    model = model_dict["model"]
    features_array = np.array(features).reshape(1, -1)
    return model.predict(features_array)[0]
