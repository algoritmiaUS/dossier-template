#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/math/gcd.h"

const int MAXN = 1e9;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // Usamos mt19937 para una generación aleatoria robusta
    mt19937 rd(clock());

    cout << "Iniciando stress test para GCD (C++)..." << endl;

    // Realizamos múltiples tests para asegurar consistencia
    // Generar 30 pares aleatorios de números entre 1 y 1e9
    int num_tests = 30;
    
    for (int t = 1; t <= num_tests; t++) {
        // Generar dos números aleatorios en rango [1, MAXN)
        int a = rd() % MAXN + 1;
        int b = rd() % MAXN + 1;
        
        // Comparar nuestra implementación gcd(a,b) con la standard __gcd(a,b)
        // Ambas deben calcular el máximo común divisor idénticamente
        assert(__gcd(a, b) == gcd(a, b));
        
    }

    cout << "¡Todos los tests de C++ para GCD pasaron con éxito!" << endl;
    return 0;
}