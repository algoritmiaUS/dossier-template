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

    vector<vector<ll>> m = {
        {0, 3, inf, 7},
        {8, 0, 2, inf},
        {5, inf, 0, 1},
        {2, inf, inf, 0}
    };

    floydWarshall(m);

    assert(m[0][2] == 5);
    assert(m[0][3] == 6);
    assert(m[3][1] == 5);

    cout << "¡Todos los tests de C++ para Floyd-Warshall pasaron con éxito!" << endl;
    return 0;
}