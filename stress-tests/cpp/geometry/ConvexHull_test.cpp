#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef long long ll;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/geometry/ConvexHull.h"

static vector<P> sortedHull(vector<P> pts) {
    sort(all(pts));
    return pts;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Convex Hull (C++)..." << endl;
    
    // TEST 1: Cuadrado con puntos interiores - el casco debe ser solo los 4 vértices
    {
        vector<P> pts = {P(0, 0), P(2, 0), P(2, 2), P(0, 2), P(1, 1), P(1, 0)};
        auto hull = convexHull(pts);
        auto got = sortedHull(hull);
        auto exp = sortedHull({P(0, 0), P(2, 0), P(2, 2), P(0, 2)});
        assert(got == exp);
    }
    
    // TEST 2: Puntos colineales - casco = puntos extremos
    {
        vector<P> pts = {P(0, 0), P(1, 1), P(2, 2), P(3, 3)};
        auto hull = convexHull(pts);
        auto got = sortedHull(hull);
        auto exp = sortedHull({P(0, 0), P(3, 3)});
        assert(got == exp);
    }

    // TEST 3: 50 iteraciones aleatorias - verificar validez del casco
    mt19937 rd(42);
    for (int t = 1; t <= 50; ++t) {
        vector<P> pts;
        // Generar 20 puntos aleatorios en rango [-10, 10]
        for (int i = 0; i < 20; ++i) {
            pts.push_back(P(int(rd() % 21) - 10, int(rd() % 21) - 10));
        }
        auto hull = convexHull(pts);
        assert(!hull.empty());  // El casco no puede estar vacío
        // Todos los puntos del casco deben estar en el conjunto original
        for (auto p : hull) {
            assert(find(pts.begin(), pts.end(), p) != pts.end());
        }
    }

    cout << "¡Todos los tests de C++ para convex hull pasaron con éxito!" << endl;
    return 0;
}