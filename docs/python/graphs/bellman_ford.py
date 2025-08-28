# Bellman-Ford
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
        if not changed:  # punto fijo alcanzado
            return dist, prec, False  # False -> no hay ciclo negativo

    return dist, prec, True  # True -> hay ciclo negativo
