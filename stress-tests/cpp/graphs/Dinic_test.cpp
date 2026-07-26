#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define sz(x) (int)(x).size()
typedef long long ll;
typedef vector<int> vi;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/Dinic.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Dinic (C++)..." << endl;

    Dinic dinic(4);
    dinic.addEdge(0, 1, 3);
    dinic.addEdge(0, 2, 2);
    dinic.addEdge(1, 2, 1);
    dinic.addEdge(1, 3, 2);
    dinic.addEdge(2, 3, 4);

    assert(dinic.calc(0, 3) == 5);

    cout << "¡Todos los tests de C++ para Dinic pasaron con éxito!" << endl;
    return 0;
}