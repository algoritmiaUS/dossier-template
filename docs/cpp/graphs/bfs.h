/**
 * Date: 2026-04-19
 * Description: Recorre todos los nodos de un grafo o árbol nivel por nivel.
 * Usage:
 *   vector<int> parent;
 *   vector<int> dist = bfs(adj, source, parent);
 */
#pragma once

vector<int> bfs(
    const vector<vector<int>>& adj, // adjacency list representation
    int s, // source vertex
    vector<int>& p // parent of each vertex
) {
    int n = (int)adj.size(); // number of nodes
    vector<int> d(n, -1);
    vector<bool> used(n, false);
    queue<int> q;

    p.assign(n, -1);
    q.push(s);
    used[s] = true;
    d[s] = 0;

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        for (int u : adj[v]) {
            if (!used[u]) {
                used[u] = true;
                q.push(u);
                d[u] = d[v] + 1;
                p[u] = v;
            }
        }
    }

    return d;
}

