"""
Description: Recorre todos los nodos de un grafo o árbol profundizando en cada rama antes de retroceder.
"""
def dfs_iterative(graph, start, seen):
    seen[start] = True
    to_visit = [start]

    while to_visit:
        node = to_visit.pop()
        for neighbor in graph[node]:
            if not seen[neighbor]:
                seen[neighbor] = True
                to_visit.append(neighbor)