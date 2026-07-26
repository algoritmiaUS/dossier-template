#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef long long ll;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/BellmanFord.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Bellman-Ford (C++)..." << endl;

    // Crear nodos y aristas
    // 4 nodos: 0, 1, 2, 3
    vector<Node> nodes(4);
    
    // Definir aristas: {desde, hacia, peso}
    // 0->1 (5), 1->2 (3), 0->2 (10)
    // Nodo 3 es inalcanzable
    vector<Ed> eds = {
        {0, 1, 5},     // Arista 0 a 1 con peso 5
        {1, 2, 3},     // Arista 1 a 2 con peso 3
        {0, 2, 10}     // Arista 0 a 2 con peso 10
    };

    // Ejecutar Bellman-Ford desde nodo 0
    bellmanFord(nodes, eds, 0);

    // Verificar distancias:
    assert(nodes[0].dist == 0);        // Nodo 0: distancia 0 (origen)
    assert(nodes[1].dist == 5);        // Nodo 1: distancia 5 (camino 0->1)
    assert(nodes[2].dist == 8);        // Nodo 2: distancia 8 (camino 0->1->2 = 5+3 < 10)
    assert(nodes[3].dist == inf);      // Nodo 3: distancia infinita (inalcanzable)

    cout << "¡Todos los tests de C++ para Bellman-Ford pasaron con éxito!" << endl;
    return 0;
}