"""
Description: Calcula los caminos más cortos desde $s$ en un grafo que puede tener aristas con pesos negativos.
Los nodos inalcanzables obtienen `dist = inf`; los nodos alcanzables a través de ciclos de peso negativo obtienen `dist = -inf`.
Se asume que $V^2 dot.op max|w_i | < 2^63$.
"""
def bellman_ford(graph, weight, source=0):
    n = len(graph)
    dist = [float('inf')] * n
    prec = [None] * n
    dist[source] = 0

    for _ in range(n):
        changed = False
        for node in range(n):
            for neighbor in graph[node]:
                alt = dist[node] + weight[node][neighbor]
                if alt < dist[neighbor]:
                    dist[neighbor] = alt
                    prec[neighbor] = node
                    changed = True
        # punto fijo alcanzado
        if not changed:  
            return dist, prec, False  # False -> no hay ciclo negativo

    return dist, prec, True  # True -> hay ciclo negativo
