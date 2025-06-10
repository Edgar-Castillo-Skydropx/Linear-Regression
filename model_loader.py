from model import linear_regression
from typing import List, Tuple, Dict
import json


def get_int_input(prompt: str) -> int:
    """Obtiene entrada entera del usuario con validación."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("❌ Por favor ingrese un número entero válido.")


def get_float_input(prompt: str) -> float:
    """Obtiene entrada flotante del usuario con validación."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("❌ Por favor ingrese un número válido.")


def get_yes_no_input(prompt: str) -> bool:
    """Obtiene respuesta sí/no del usuario."""
    while True:
        response = input(prompt).upper().strip()
        if response in ["S", "SI", "SÍ", "Y", "YES"]:
            return True
        elif response in ["N", "NO"]:
            return False
        else:
            print("❌ Por favor responda 'S' para sí o 'N' para no.")


def save_model_info(
    concept: str,
    variable_names: List[str],
    model_info: Dict,
    filename: str = "trained_model.json",
):
    """Guarda la información del modelo en un archivo JSON."""
    try:
        # Convertir arrays numpy a listas para serialización JSON
        save_data = {
            "concept": concept,
            "variable_names": variable_names,
            "intercept": float(model_info["intercept"]),
            "coef": (
                model_info["coef"].tolist()
                if hasattr(model_info["coef"], "tolist")
                else list(model_info["coef"])
            ),
            "r2_score": float(model_info["r2_score"]),
            "rmse": float(model_info["rmse"]),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Modelo guardado en {filename}")
        return True

    except Exception as e:
        print(f"❌ Error al guardar el modelo: {e}")
        return False


def collect_training_data() -> Tuple[str, List[str], List[List[float]], List[float]]:
    """
    Recolecta datos de entrenamiento del usuario.

    Returns:
        Tuple con: concepto, nombres de variables, datos X, valores y
    """
    print("🤖 Configuración del Modelo de Regresión Lineal")
    print("=" * 50)

    # Obtener información básica
    concept = input("¿Qué deseas predecir? ")
    num_variables = get_int_input("¿Cuántas variables predictoras tienes? ")

    # Obtener nombres de variables
    variable_names = []
    for i in range(num_variables):
        name = input(f"Nombre de la variable #{i+1}: ")
        variable_names.append(name)

    print(f"\n📊 Ingreso de Datos de Entrenamiento para '{concept}'")
    print("-" * 50)

    # Recolectar datos de entrenamiento
    X_data = []
    y_data = []
    sample_count = 0

    while True:
        sample_count += 1
        print(f"\n📝 Muestra #{sample_count}")

        # Recolectar variables predictoras
        sample_features = []
        for var_name in variable_names:
            value = get_float_input(f"  {var_name}: ")
            sample_features.append(value)

        # Recolectar variable objetivo
        target_value = get_float_input(f"  {concept} (valor real): ")

        X_data.append(sample_features)
        y_data.append(target_value)

        # Preguntar si continuar
        if not get_yes_no_input("¿Desea ingresar más datos? (S/N): "):
            break

    print(f"\n✅ Se recolectaron {len(X_data)} muestras de entrenamiento.")
    return concept, variable_names, X_data, y_data


def main():
    """Función principal para entrenar el modelo."""
    try:
        # Recolectar datos
        concept, variable_names, X_data, y_data = collect_training_data()

        # Entrenar modelo
        print("\n🔄 Entrenando modelo...")
        model_info = linear_regression(X_data, y_data)

        # Mostrar resultados del entrenamiento
        print("\n📈 Resultados del Entrenamiento")
        print("=" * 40)
        print(f"Intercepto: {model_info['intercept']:.4f}")
        print("Coeficientes:")
        for i, (name, coef) in enumerate(zip(variable_names, model_info["coef"])):
            print(f"  {name}: {coef:.4f}")
        print(f"R² Score: {model_info['r2_score']:.4f}")
        print(f"RMSE: {model_info['rmse']:.4f}")

        # Guardar modelo automáticamente
        print("\n💾 Guardando modelo...")
        if save_model_info(concept, variable_names, model_info):
            print("🎉 ¡Modelo entrenado y guardado exitosamente!")
            print(
                "Ahora puedes ejecutar 'application_mejorada.py' para hacer predicciones."
            )
        else:
            print("⚠️ El modelo se entrenó pero no se pudo guardar.")

        return {
            "concept": concept,
            "variable_names": variable_names,
            "model_info": model_info,
        }

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None


if __name__ == "__main__":
    trained_model = main()
