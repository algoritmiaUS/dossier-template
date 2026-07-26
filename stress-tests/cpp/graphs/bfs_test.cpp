#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/graphs/bfs.h"

static void run_bfs_chain_test() {
    vector<vector<int>> adj = {
        {1},
        {2},
        {3},
        {}
    };
    vector<int> parent;
    vector<int> dist = bfs(adj, 0, parent);

    assert(dist == vector<int>({0, 1, 2, 3}));
    assert(parent == vector<int>({-1, 0, 1, 2}));
}

static void run_bfs_branch_test() {
    vector<vector<int>> adj = {
        {1, 2},
        {3},
        {3},
        {}
    };
    vector<int> parent;
    vector<int> dist = bfs(adj, 0, parent);

    assert(dist == vector<int>({0, 1, 1, 2}));
    assert(parent[1] == 0);
    assert(parent[2] == 0);
    assert(parent[3] == 1 || parent[3] == 2);
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para BFS (C++)..." << endl;
    run_bfs_chain_test();
    run_bfs_branch_test();
    cout << "¡Todos los tests de C++ para BFS pasaron con éxito!" << endl;
    return 0;
}
