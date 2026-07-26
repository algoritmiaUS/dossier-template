#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/Dijkstra.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Dijkstra (C++)..." << endl;

    adj = {
        {{1, 2}, {2, 5}},
        {{2, 1}, {3, 4}},
        {{3, 1}},
        {}
    };

    vector<int> d, p;
    dijkstra(0, d, p);

    assert(d == vector<int>({0, 2, 3, 4}));
    assert(p == vector<int>({-1, 0, 1, 2}));

    cout << "¡Todos los tests de C++ para Dijkstra pasaron con éxito!" << endl;
    return 0;
}