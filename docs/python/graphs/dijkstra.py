"""
Description: Se utiliza para encontrar el camino más corto desde un nodo de inicio hasta
todos los demás nodos en un grafo ponderado.
"""
from heapq import heappush, heappop
def dijkstra(
    al: List[List[Tuple[int, int]]], s: int,
    s: int,
):
    dist = [float("inf")] * len(al)
    dist[s] = 0
    pq = [(0, s)]
    while 0 < len(pq):
        d, u = heappop(pq) 

        if d > dist[u]:
            continue 
    
        for v, w in al[u]:
            if dist[u] + w < dist[v]:  
                dist[v] = dist[u] + w
                heappush(pq, (dist[v], v))
    
    return dist