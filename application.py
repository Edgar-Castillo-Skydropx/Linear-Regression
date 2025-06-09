from typing import List
from model_loader import model, concept, options

respuesta: str = "S"

while respuesta == "S":
    print("¿Desea realizar predicción? (S/N)")
    respuesta = input().upper()
    if respuesta == "N":
        break
    elif respuesta == "S":
        current_options: List[int] = []
        for option in options:
            current_options.append(
                int(input(f"Ingresa la variable: {option}, del concepto: {concept}\n"))
            )

        value = float(model["intercept"])
        for i, coef in enumerate(model["coef"]):
            value += coef * current_options[i]

        print(f"El valor estimado de: {concept} es: {value:.2f} unidades.")

    else:
        print("Respuesta no válida. Por favor ingrese 'S' o 'N'.")
