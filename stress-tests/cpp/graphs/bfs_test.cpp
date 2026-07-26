#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/bfs.h"

static void run_bfs_chain_test() {
    // TEST 1: Grafo lineal 0 -> 1 -> 2 -> 3
    vector<vector<int>> adj = {
        {1},    // Nodo 0 conecta a 1
        {2},    // Nodo 1 conecta a 2
        {3},    // Nodo 2 conecta a 3
        {}      // Nodo 3 es hoja
    };
    vector<int> parent;
    vector<int> dist = bfs(adj, 0, parent);

    // Verificar distancias desde nodo 0: 0->0 es 0, 0->1 es 1, etc.
    assert(dist == vector<int>({0, 1, 2, 3}));
    // Verificar árbol de padres
    assert(parent == vector<int>({-1, 0, 1, 2}));
}

static void run_bfs_branch_test() {
    // TEST 2: Grafo con bifurcación: 0 -> [1, 2], 1 -> 3, 2 -> 3
    vector<vector<int>> adj = {
        {1, 2},  // Nodo 0 conecta a 1 y 2
        {3},     // Nodo 1 conecta a 3
        {3},     // Nodo 2 conecta a 3
        {}       // Nodo 3 es hoja
    };
    vector<int> parent;
    vector<int> dist = bfs(adj, 0, parent);

    // Verificar distancias
    assert(dist == vector<int>({0, 1, 1, 2}));
    // Verificar que el árbol BFS es válido
    assert(parent[1] == 0);      // Padre de 1 es 0
    assert(parent[2] == 0);      // Padre de 2 es 0
    assert(parent[3] == 1 || parent[3] == 2);  // Padre de 3 puede ser 1 o 2
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para BFS (C++)..." << endl;
    run_bfs_chain_test();
    run_bfs_branch_test();
    cout << "¡Todos los tests de C++ para BFS pasaron con éxito!" << endl;
    return 0;
}
