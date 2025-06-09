import model as LinearModel
from typing import List

concept: str = input("¿Que deseas predecir?\n")
_properties: int = int(input("¿Cuantas variables tienes?\n"))

options: List[str] = []

for i in range(_properties):
    _option: str = input(f"Nombre de la variable: #{i+1}\n")
    options.append(_option)

breakW = "N"

i: int = 0
data: List[int] = []
values: List[int] = []
while breakW == "N":
    i += 1
    print(f"Datos de Entrenamiento, #{i}")
    _dataI: List[int] = []
    for option in options:
        if option != options[len(options) - 1]:
            _dataI.append(int(input(f"{option}: ")))
        else:
            values.append(int(input(f"{option}: ")))
    data.append(_dataI)
    breakW = input("¿Desea ingresar más datos? (S/N)\n").upper()


# [[80, 2, 5], [100, 3, 10], [120, 4, 2], [90, 3, 8], [110, 3, 4]],  [1300, 1500, 2000, 1400, 1800]
model = LinearModel.linear_regression(data, values)
