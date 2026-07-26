#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef long long ll;
typedef vector<int> vi;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/EdmondsKarp.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Edmonds-Karp (C++)..." << endl;

    // Crear grafo de flujo con 4 nodos (usando diccionarios)
    // 0 = fuente, 3 = sumidero
    vector<unordered_map<int, ll>> graph(4);
    
    // Agregar aristas con capacidades
    graph[0][1] = 3;  // 0 -> 1: capacidad 3
    graph[0][2] = 2;  // 0 -> 2: capacidad 2
    graph[1][2] = 1;  // 1 -> 2: capacidad 1
    graph[1][3] = 2;  // 1 -> 3: capacidad 2
    graph[2][3] = 4;  // 2 -> 3: capacidad 4

    // Calcular flujo máximo usando Edmonds-Karp (BFS-based)
    // Mismo resultado que Dinic: flujo máximo = 5
    assert(edmondsKarp(graph, 0, 3) == 5);

    cout << "¡Todos los tests de C++ para Edmonds-Karp pasaron con éxito!" << endl;
    return 0;
}