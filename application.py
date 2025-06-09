from typing import List
from model_loader import model

respuesta: str = "S"

while respuesta == "S":
    print("¿Desea realizar predicción? (S/N)")
    respuesta = input().upper()
    if respuesta == "N":
        break
    elif respuesta == "S":
        options: List[int] = []
        print("Ingresa los metros cuadrados de la casa:")
        options.append(int(input()))
        print("Ingresa el número de habitaciones:")
        options.append(int(input()))
        print("Ingresa los años de antiguedad:")
        options.append(int(input()))

        value = float(model["intercept"])
        for i, coef in enumerate(model["coef"]):
            value += coef * options[i]

        print(f"El valor estimado de la casa es: {value:.2f} unidades monetarias")

    else:
        print("Respuesta no válida. Por favor ingrese 'S' o 'N'.")
