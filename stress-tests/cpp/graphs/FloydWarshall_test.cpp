#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define sz(x) (int)(x).size()
typedef long long ll;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/FloydWarshall.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Floyd-Warshall (C++)..." << endl;

    // Crear matriz de distancias iniciales (inf significa sin arista directa)
    // Matriz 4x4: 0->1(3), 0->3(7), 1->0(8), 1->2(2)
    //             2->0(5), 2->3(1), 3->0(2)
    vector<vector<ll>> m = {
        {0, 3, inf, 7},     // Desde nodo 0
        {8, 0, 2, inf},     // Desde nodo 1
        {5, inf, 0, 1},     // Desde nodo 2
        {2, inf, inf, 0}    // Desde nodo 3
    };

    // Ejecutar Floyd-Warshall para calcular distancias mínimas
    floydWarshall(m);

    // Verificar caminos mínimos calculados:
    assert(m[0][2] == 5);   // 0->2: mejor camino es 0->1->2 (3+2=5)
    assert(m[0][3] == 6);   // 0->3: mejor camino es 0->1->2->3 (3+2+1=6)
    assert(m[3][1] == 5);   // 3->1: mejor camino es 3->0->1 (2+3=5)

    cout << "¡Todos los tests de C++ para Floyd-Warshall pasaron con éxito!" << endl;
    return 0;
}