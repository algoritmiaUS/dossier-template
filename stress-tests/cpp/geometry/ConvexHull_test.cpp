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
    {
        vector<P> pts = {P(0, 0), P(2, 0), P(2, 2), P(0, 2), P(1, 1), P(1, 0)};
        auto hull = convexHull(pts);
        auto got = sortedHull(hull);
        auto exp = sortedHull({P(0, 0), P(2, 0), P(2, 2), P(0, 2)});
        assert(got == exp);
    }
    {
        vector<P> pts = {P(0, 0), P(1, 1), P(2, 2), P(3, 3)};
        auto hull = convexHull(pts);
        auto got = sortedHull(hull);
        auto exp = sortedHull({P(0, 0), P(3, 3)});
        assert(got == exp);
    }

    mt19937 rd(42);
    for (int t = 1; t <= 50; ++t) {
        vector<P> pts;
        for (int i = 0; i < 20; ++i) {
            pts.push_back(P(int(rd() % 21) - 10, int(rd() % 21) - 10));
        }
        auto hull = convexHull(pts);
        assert(!hull.empty());
        for (auto p : hull) {
            assert(find(pts.begin(), pts.end(), p) != pts.end());
        }
    }

    cout << "¡Todos los tests de C++ para convex hull pasaron con éxito!" << endl;
    return 0;
}