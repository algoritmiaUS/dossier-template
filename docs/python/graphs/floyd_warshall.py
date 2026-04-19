"""
Description:Calcula las distancias más cortas entre todos los pares de un grafo dirigido que podría tener aristas con pesos negativos.
La entrada es una matriz de distancias m, donde m[i][j] = inf si i y j no son adyacentes.
Como salida, m[i][j] se establece en la distancia más corta entre i y j, inf si no existe camino, o -inf si el camino pasa por un ciclo de peso negativo.
"""
import sys
input = sys.stdin.readline
def leer_grafo_floyd(n: int, e: int):
    am = [
        [float("inf") for _ in range(n)]
        for _ in range(n)
    ]
    for u in range(n):
        am[u][u] = 0
    for _ in range(e):
        u, v, w = map(int, input().split())
        # Se guarda la menor distancia si hay
        # aristas repetidas
        am[u][v] = min(am[u][v], w)
    return am


def floyd_warshall(am: list[list[int]], n: int):
    """Careful! This modifies am."""

    for k in range(n):
        for u in range(n): 
            for v in range(n):
                am[u][v] = min(
                    am[u][v],
                    am[u][k] + am[k][v]
                )
    return am

    