#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/math/sieve.h"

const int MAXN = 1e6;

// Función ingenua (fuerza bruta) para comprobar si es primo
// Complejidad: O(sqrt(N))
bool is_prime_naive(int x) {
    if (x < 2) return false;
    for (int i = 2; i * i <= x; i++) {
        if (x % i == 0) return false;
    }
    return true;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // Usamos mt19937 para una generación aleatoria robusta
    mt19937 rd(clock());

    cout << "Iniciando stress test para Criba de Eratóstenes (C++)..." << endl;

    // Realizamos múltiples tests para asegurar consistencia
    int num_tests = 30; 
    
    for (int t = 1; t <= num_tests; t++) {
        int n = rd() % MAXN + 1;
        
        // Llamamos a tu implementación
        criba(n);
        
        // 1. Verificamos el vector booleano 'es_primo'
        int primes_expected = 0;
        for (int i = 2; i <= n; i++) {
            bool expected = is_prime_naive(i);
            // Comparamos tu criba con la función ingenua
            assert(es_primo[i] == expected);
            
            if (expected) {
                primes_expected++;
            }
        }
        
        // 2. Verificamos el tamaño del vector 'primos'
        assert((int)primos.size() == primes_expected);
        
        // 3. Verificamos el contenido y el orden del vector 'primos'
        for (size_t i = 0; i < primos.size(); i++) {
            assert(is_prime_naive(primos[i]));
            // Aseguramos que la lista sea estrictamente creciente
            if (i > 0) {
                assert(primos[i] > primos[i - 1]);
            }
        }
        
    }

    cout << "¡Todos los tests de C++ para Criba de Eratóstenes pasaron con éxito!" << endl;
    return 0;
}