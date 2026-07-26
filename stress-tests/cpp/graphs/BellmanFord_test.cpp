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

    vector<Node> nodes(4);
    vector<Ed> eds = {
        {0, 1, 5},
        {1, 2, 3},
        {0, 2, 10}
    };

    bellmanFord(nodes, eds, 0);

    assert(nodes[0].dist == 0);
    assert(nodes[1].dist == 5);
    assert(nodes[2].dist == 8);
    assert(nodes[3].dist == inf);

    cout << "¡Todos los tests de C++ para Bellman-Ford pasaron con éxito!" << endl;
    return 0;
}