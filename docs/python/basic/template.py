import sys

# Ajustar el límite de recursión
sys.setrecursionlimit(10**6)

"""
Descomenta para usar sys.stdin.readline en lugar de
input() y acelerar la lectura de datos. Advertencia:
sys.stdin.readline incluye un salto de línea (\n), por
lo que debes usar .strip() para eliminarlo.

Por ejemplo:
Leer una línea de entrada y eliminar el salto de línea
Usamos strip() para eliminar el salto de línea 
n = int(input().strip()) 

"""
# input = sys.stdin.readline

def leer_un_numero():
    return int(input())

def leer_varios_numeros():
    return list(
        map(int, input().split())
    )

def main():
    # Código principal aquí
    pass

if __name__ == "__main__":

    main()