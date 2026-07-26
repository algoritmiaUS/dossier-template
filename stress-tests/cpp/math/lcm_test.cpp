#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

#include "../../../docs/cpp/math/gcd.h"
// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/math/lcm.h"

const int MAXN = 1e9;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // Usamos mt19937 para una generación aleatoria robusta
    mt19937 rd(clock());

    cout << "Iniciando stress test para lcm (C++)..." << endl;

    // Realizamos múltiples tests para asegurar consistencia
    // Generar 30 pares aleatorios de números entre 1 y 1e9
    int num_tests = 30;
    
    for (int t = 1; t <= num_tests; t++) {
        // Generar dos números aleatorios en rango [1, MAXN)
        int a = rd() % MAXN + 1;
        int b = rd() % MAXN + 1;

        // Verificar lcm(a,b) usando la fórmula: lcm(a,b) = (a*b)/gcd(a,b)
        // Usamos (long long) para evitar overflow en la multiplicación a*b
        // Ambas deben calcular el mínimo común múltiplo idénticamente
        assert(((long long)a*b)/__gcd(a, b) == lcm(a, b));
        
    }

    cout << "¡Todos los tests de C++ para lcm pasaron con éxito!" << endl;
    return 0;
}