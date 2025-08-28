# Cuando no sabemos cuántas líneas se leerán desde la entrada estándar y queremos procesarlas una por una hasta que se agoten, una forma común en Python es utilizar un bucle infinito y capturar la excepción EOFError.

while True:
    try:
        x, y = map(int, input().split())
        print(abs(x - y))
    except EOFError:
        break

# Otro enfoque sería leer toda la entrada de una vez antes de comenzar a procesarla. Esto es adecuado cuando la entrada no es muy grande.

import sys

for line in sys.stdin.readlines():
     [x, y] = list(map(int, line.split()))
     print(abs(x - y))
