#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/dfs.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para DFS (C++)..." << endl;

    // TEST 1: Verificar que DFS alcanza todos los nodos conectados
    // Grafo: 0 -> [1, 2], 1 -> 3, 2 -> 4 (árbol conexo)
    adj = {
        {1, 2},  // Nodo 0 conecta a 1 y 2
        {3},     // Nodo 1 conecta a 3
        {4},     // Nodo 2 conecta a 4
        {},      // Nodo 3 es hoja
        {}       // Nodo 4 es hoja
    };
    n = (int)adj.size();
    visited.assign(n, false);

    // Ejecutar DFS desde nodo 0
    dfs(0);

    // Verificar que todos los nodos fueron visitados (1 = alcanzable desde 0)
    for (int i = 0; i < n; ++i) {
        assert(visited[i]);
    }

    // TEST 2: Verificar que DFS NO alcanza nodos desconectados
    // Agregar un nodo 5 aislado (no conectado)
    adj.push_back({});
    n = (int)adj.size();
    visited.assign(n, false);
    dfs(0);
    
    // El nodo 5 no debe ser visitado porque no es alcanzable desde 0
    assert(!visited[5]);

    cout << "¡Todos los tests de C++ para DFS pasaron con éxito!" << endl;
    return 0;
}