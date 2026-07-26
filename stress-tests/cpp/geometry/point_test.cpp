#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/geometry/point.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para Point (C++)..." << endl;

    // 1. DEFINIR PUNTOS: Crear puntos de prueba con coordenadas enteras
    using P = Point<long long>;
    P a(3, 4), b(-1, 2), c(2, 6);

    // 2. OPERACIONES ARITMÉTICAS: Suma, resta, multiplicación escalar
    assert(a + b == P(2, 6));        // Suma: (3+(-1), 4+2) = (2, 6)
    assert(a - b == P(4, 2));        // Resta: (3-(-1), 4-2) = (4, 2)
    assert(a * 2 == P(6, 8));        // Escalar: (3*2, 4*2) = (6, 8)
    
    // 3. PRODUCTOS: Producto punto y producto cruz
    assert(a.dot(b) == 5);           // Dot: 3*(-1) + 4*2 = -3+8 = 5
    assert(a.cross(b) == 10);        // Cross 2D: 3*2 - 4*(-1) = 6+4 = 10
    assert(a.cross(b, c) == (b - a).cross(c - a)); // Cross product verificación
    
    // 4. DISTANCIAS Y SIGNO: Distancia al cuadrado y función signo
    assert(a.dist2() == 25);         // Distance² = 3² + 4² = 9+16 = 25
    assert(sgn(-5) == -1);           // Signo negativo
    assert(sgn(0) == 0);             // Signo cero
    assert(sgn(7) == 1);             // Signo positivo

    // 5. ROTACIÓN: Rotar un punto 90 grados
    P p(1, 0);
    P q = p.rotate(M_PI / 2);
    assert(q == P(0, 1));            // (1,0) rotado 90° = (0,1)

    cout << "¡Todos los tests de C++ para point pasaron con éxito!" << endl;
    return 0;
}