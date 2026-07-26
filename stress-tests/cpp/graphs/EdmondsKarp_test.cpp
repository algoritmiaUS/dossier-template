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

    vector<unordered_map<int, ll>> graph(4);
    graph[0][1] = 3;
    graph[0][2] = 2;
    graph[1][2] = 1;
    graph[1][3] = 2;
    graph[2][3] = 4;

    assert(edmondsKarp(graph, 0, 3) == 5);

    cout << "¡Todos los tests de C++ para Edmonds-Karp pasaron con éxito!" << endl;
    return 0;
}