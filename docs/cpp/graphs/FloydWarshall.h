/**
 * Date: 2026-04-19
 * Description: Calcula las distancias más cortas entre todos los pares de un grafo dirigido que podría tener aristas con pesos negativos.
 * La entrada es una matriz de distancias m, donde m[i][j] = inf si i y j no son adyacentes.
 * Como salida, m[i][j] se establece en la distancia más corta entre i y j, inf si no existe camino,
 * o -inf si el camino pasa por un ciclo de peso negativo.
 */
#pragma once

const ll inf = 1LL << 62;
void floydWarshall(vector<vector<ll>>& m) {
	int n = sz(m);
	rep(i,0,n) m[i][i] = min(m[i][i], 0LL);
	rep(k,0,n) rep(i,0,n) rep(j,0,n)
		if (m[i][k] != inf && m[k][j] != inf) {
			auto newDist = max(m[i][k] + m[k][j], -inf);
			m[i][j] = min(m[i][j], newDist);
		}
	rep(k,0,n) if (m[k][k] < 0) rep(i,0,n) rep(j,0,n)
		if (m[i][k] != inf && m[k][j] != inf) m[i][j] = -inf;
}