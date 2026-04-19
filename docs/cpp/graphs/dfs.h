/**
 * Date: 2026-04-19
 * Description: Recorre todos los nodos de un grafo o árbol profundizando en cada rama antes de retroceder.
 */
vector<vector<int>> adj; // graph represented as an adjacency list
int n; // number of vertices

vector<bool> visited;

void dfs(int v) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u])
            dfs(u);
    }
}