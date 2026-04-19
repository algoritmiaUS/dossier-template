def twosat(n, clauses):
    g = [[] for _ in range(2 * n)]
    rg = [[] for _ in range(2 * n)]

    def lit(i, val):
        return 2 * i + (0 if val else 1)

    for i, vi, j, vj in clauses:
        a = lit(i, vi)
        b = lit(j, vj)
        g[a ^ 1].append(b)
        g[b ^ 1].append(a)
        rg[b].append(a ^ 1)
        rg[a].append(b ^ 1)

    order = []
    seen = [0] * (2 * n)

    def dfs(u):
        seen[u] = 1
        for v in g[u]:
            if not seen[v]:
                dfs(v)
        order.append(u)

    comp = [-1] * (2 * n)

    def rdfs(u, c):
        comp[u] = c
        for v in rg[u]:
            if comp[v] == -1:
                rdfs(v, c)

    for u in range(2 * n):
        if not seen[u]:
            dfs(u)
    c = 0
    for u in order[::-1]:
        if comp[u] == -1:
            rdfs(u, c)
            c += 1
    ans = [0] * n
    for i in range(n):
        if comp[2 * i] == comp[2 * i + 1]:
            return None
        ans[i] = comp[2 * i] > comp[2 * i + 1]
    return ans
