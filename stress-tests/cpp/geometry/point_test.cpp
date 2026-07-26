#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/geometry/point.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Point (C++)..." << endl;

    using P = Point<long long>;

    P a(3, 4), b(-1, 2), c(2, 6);

    assert(a + b == P(2, 6));
    assert(a - b == P(4, 2));
    assert(a * 2 == P(6, 8));
    assert(a.dot(b) == 5);
    assert(a.cross(b) == 10);
    assert(a.cross(b, c) == (b - a).cross(c - a));
    assert(a.dist2() == 25);
    assert(sgn(-5) == -1);
    assert(sgn(0) == 0);
    assert(sgn(7) == 1);

    P p(1, 0);
    P q = p.rotate(M_PI / 2);
    assert(q == P(0, 1));

    cout << "¡Todos los tests de C++ para point pasaron con éxito!" << endl;
    return 0;
}