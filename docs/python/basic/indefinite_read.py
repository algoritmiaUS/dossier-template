"""
Description: Lectura eficiente hasta el final de la entrada (EOF). Procesa línea a línea
"""
import sys
for line in sys.stdin:
    x, y = map(int, line.split())
    print(abs(x - y))

