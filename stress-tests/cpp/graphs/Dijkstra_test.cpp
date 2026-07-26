#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/Dijkstra.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Dijkstra (C++)..." << endl;

    // Construir grafo ponderado con 4 nodos
    // Aristas: 0->1 (peso 2), 0->2 (peso 5)
    //          1->2 (peso 1), 1->3 (peso 4)
    //          2->3 (peso 1)
    adj = {
        {{1, 2}, {2, 5}},  // Nodo 0: aristas a 1 (2) y 2 (5)
        {{2, 1}, {3, 4}},  // Nodo 1: aristas a 2 (1) y 3 (4)
        {{3, 1}},          // Nodo 2: arista a 3 (1)
        {}                 // Nodo 3: sin aristas salientes
    };

    // Vectores para guardar distancias y predecesores
    vector<int> d, p;
    
    // Ejecutar Dijkstra desde nodo 0
    dijkstra(0, d, p);

    // Verificar distancias mínimas desde 0:
    // 0: 0 (origen)
    // 1: 2 (0->1)
    // 2: 3 (0->1->2)
    // 3: 4 (0->1->3)
    assert(d == vector<int>({0, 2, 3, 4}));
    
    // Verificar árbol de caminos mínimos (padres en el árbol):
    // 0: -1 (raíz), 1: 0, 2: 1, 3: 2
    assert(p == vector<int>({-1, 0, 1, 2}));

    cout << "¡Todos los tests de C++ para Dijkstra pasaron con éxito!" << endl;
    return 0;
}