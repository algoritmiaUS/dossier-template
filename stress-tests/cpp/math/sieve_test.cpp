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
    // Generar 30 números aleatorios en [1, 1e6) y verificar criba en cada uno
    int num_tests = 30;
    
    for (int t = 1; t <= num_tests; t++) {
        int n = rd() % MAXN + 1;
        
        // Callable la criba para calcular todos los primos hasta n
        criba(n);
        
        // TEST 1: Verificar el vector booleano 'es_primo'
        // Comparar cada elemento es_primo[i] con verificación naïve is_prime_naive(i)
        // Ambas funciones deben coincidir completamente en determinar primalidad
        int primes_expected = 0;
        for (int i = 2; i <= n; i++) {
            bool expected = is_prime_naive(i);
            assert(es_primo[i] == expected);  // Validar coherencia bit a bit
            
            if (expected) {
                primes_expected++;  // Contar primos encontrados
            }
        }
        
        // TEST 2: Verificar el tamaño del vector 'primos'
        // El vector de primos debe contener exactamente 'primes_expected' elementos
        // Esto valida que la criba no omitió ni agregó primos incorrectamente
        assert((int)primos.size() == primes_expected);
        
        // TEST 3: Verificar contenido y orden del vector 'primos'
        // Para cada primo en el vector:
        // - Verificar que es efectivamente primo (validación con is_prime_naive)
        // - Verificar que está en orden estrictamente creciente
        for (size_t i = 0; i < primos.size(); i++) {
            assert(is_prime_naive(primos[i]));     // Cada elemento debe ser primo
            if (i > 0) {
                assert(primos[i] > primos[i - 1]); // Orden creciente estricto
            }
        }
        
    }

    cout << "¡Todos los tests de C++ para Criba de Eratóstenes pasaron con éxito!" << endl;
    return 0;
}