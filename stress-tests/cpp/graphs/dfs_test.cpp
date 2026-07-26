#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/dfs.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para DFS (C++)..." << endl;

    adj = {
        {1, 2},
        {3},
        {4},
        {},
        {}
    };
    n = (int)adj.size();
    visited.assign(n, false);

    dfs(0);

    for (int i = 0; i < n; ++i) {
        assert(visited[i]);
    }

    adj.push_back({});
    n = (int)adj.size();
    visited.assign(n, false);
    dfs(0);
    assert(!visited[5]);

    cout << "¡Todos los tests de C++ para DFS pasaron con éxito!" << endl;
    return 0;
}