def leer_grafo_dirigido_ponderado(V, E):
    """Lee un grafo dirigido y ponderado."""
    AL = [[] for _ in range(V)]
    for _ in range(E):
        u, v, w = map(int, sys.stdin.readline().split())
        AL[u].append((v, w))  # Solo dirección u -> v con peso w
    return AL

def leer_grafo_dirigido_no_ponderado(V, E):
    """Lee un grafo dirigido y no ponderado."""
    AL = [[] for _ in range(V)]
    for _ in range(E):
        u, v = map(int, sys.stdin.readline().split())
        AL[u].append(v)  # Solo dirección u -> v sin peso
    return AL

def leer_grafo_no_dirigido_ponderado(V, E):
    """Lee un grafo no dirigido y ponderado."""
    AL = [[] for _ in range(V)]
    for _ in range(E):
        u, v, w = map(int, sys.stdin.readline().split())
        AL[u].append((v, w))  # u -> v con peso w
        AL[v].append((u, w))  # v -> u con peso w
    return AL

def leer_grafo_no_dirigido_no_ponderado(V, E):
    """Lee un grafo no dirigido y no ponderado."""
    AL = [[] for _ in range(V)]
    for _ in range(E):
        u, v = map(int, sys.stdin.readline().split())
        AL[u].append(v)  # u -> v sin peso
        AL[v].append(u)  # v -> u sin peso
    return AL