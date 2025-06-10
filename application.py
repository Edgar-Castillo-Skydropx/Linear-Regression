from model import predict_value
from model_loader import get_float_input, get_yes_no_input
from typing import Dict, List
import json


def load_model_info(filename: str = "trained_model.json") -> Dict:
    """Carga la información del modelo desde un archivo JSON."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo {filename}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return None


def predict_with_equation(model_data: Dict, features: List[float]) -> float:
    """
    Realiza una predicción usando la ecuación de regresión lineal.

    Predicción = intercept + coef[0]*feature[0] + coef[1]*feature[1] + ...
    """
    intercept = model_data["intercept"]
    coefficients = model_data["coef"]

    prediction = intercept
    for i, (coef, feature) in enumerate(zip(coefficients, features)):
        prediction += coef * feature

    return prediction


def make_predictions(model_data: Dict):
    """Realiza predicciones usando el modelo entrenado."""
    concept = model_data["concept"]
    variable_names = model_data["variable_names"]

    print(f"\n🔮 Predicciones para '{concept}'")
    print("=" * 50)

    while True:
        print(f"\n📝 Ingrese los valores para predecir '{concept}':")

        # Recolectar valores para predicción
        features = []
        for var_name in variable_names:
            value = get_float_input(f"  {var_name}: ")
            features.append(value)

        # Realizar predicción
        try:
            prediction = predict_with_equation(model_data, features)

            print(f"\n🎯 Predicción:")
            print(f"  {concept}: {prediction:.2f} unidades")

            # Mostrar ecuación utilizada
            print(f"\n📐 Ecuación utilizada:")
            equation = f"  {concept} = {model_data['intercept']:.4f}"
            for i, (name, coef) in enumerate(zip(variable_names, model_data["coef"])):
                sign = "+" if coef >= 0 else ""
                equation += f" {sign}{coef:.4f}*{name}"
            print(equation)

        except Exception as e:
            print(f"❌ Error en la predicción: {e}")

        # Preguntar si continuar
        if not get_yes_no_input("\n¿Desea realizar otra predicción? (S/N): "):
            break


def show_model_summary(model_data: Dict):
    """Muestra un resumen del modelo entrenado."""
    concept = model_data["concept"]
    variable_names = model_data["variable_names"]

    print(f"\n📊 Resumen del Modelo")
    print("=" * 40)
    print(f"Concepto predicho: {concept}")
    print(f"Variables predictoras: {', '.join(variable_names)}")
    print(f"Precisión (R²): {model_data['r2_score']:.4f}")
    print(f"Error (RMSE): {model_data['rmse']:.4f}")

    # Interpretación del R²
    r2 = model_data["r2_score"]
    if r2 >= 0.9:
        interpretation = "Excelente"
    elif r2 >= 0.7:
        interpretation = "Buena"
    elif r2 >= 0.5:
        interpretation = "Moderada"
    else:
        interpretation = "Pobre"

    print(f"Calidad del modelo: {interpretation}")


def main_application():
    """Aplicación principal para realizar predicciones."""
    print("🤖 Sistema de Predicción con Regresión Lineal")
    print("=" * 50)

    # Intentar cargar modelo existente
    model_data = load_model_info()

    if model_data is None:
        print("No se encontró un modelo entrenado.")
        print("Por favor, ejecute primero 'model_loader.py' para entrenar un modelo.")
        return

    # Mostrar resumen del modelo
    show_model_summary(model_data)

    # Menú principal
    while True:
        print(f"\n🎯 Opciones disponibles:")
        print("1. Realizar predicciones")
        print("2. Ver resumen del modelo")
        print("3. Salir")

        try:
            option = input("\nSeleccione una opción (1-3): ").strip()

            if option == "1":
                make_predictions(model_data)
            elif option == "2":
                show_model_summary(model_data)
            elif option == "3":
                print("👋 ¡Hasta luego!")
                break
            else:
                print("❌ Opción no válida. Por favor seleccione 1, 2 o 3.")

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    main_application()
