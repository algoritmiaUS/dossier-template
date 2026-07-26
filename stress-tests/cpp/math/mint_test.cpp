#include <bits/stdc++.h>

using namespace std;
typedef long long ll;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/math/mint.h"

const int MAXA = 1e6;
const int MAXB = 1e9;

int potencia(int a, int p){
    ll sol = 1;
    while(p--) {sol *= a; sol %= MOD;}
    return sol;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // Usamos mt19937 para una generación aleatoria robusta
    mt19937 rd(clock());

    cout << "Iniciando stress test para mint (C++)..." << endl;

    // Realizamos múltiples tests para asegurar consistencia
    // Generar 30 pares aleatorios de números: a ∈ [1, 1e6), b ∈ [1, 1e9), a%MOD ≤ b%MOD
    int num_tests = 30;
    
    for (int t = 1; t <= num_tests; t++) {
        int a = rd() % MAXA + 1;
        int b = rd() % MAXB + 1;
        while(b == a) b = rd() % MAXB + 1;        // Asegurar a ≠ b
        if (a%MOD > b%MOD) swap(a, b);            // Asegurar a%MOD ≤ b%MOD

        Mint aM = a, bM = b;
        
        // TEST 1: Operador de igualdad (==)
        // Verificar que Mint(a) == a (comparación con int)
        assert(aM == a);

        // TEST 2: Operador menor que (<)
        // Verificar que aM < bM (porque a%MOD ≤ b%MOD por construcción)
        assert(aM < bM);

        // TEST 3: Operador mayor que (>)
        // Verificar que bM > aM (relación inversa)
        assert(bM > aM);

        // TEST 4: Suma modular (+)
        // Verificar que (aM + bM) = (a + b) % MOD
        assert(aM+bM == (a+b)%MOD);

        // TEST 5: Resta modular (-)
        // Verificar que (aM - bM) = (a - b) % MOD (con ajuste de signo)
        assert(aM-bM == (a-b)%MOD);

        // TEST 6: Multiplicación modular (*)
        // Verificar que (aM * bM) = (a * b) % MOD (usando long long para evitar overflow)
        assert(aM*bM == ((ll)a*b)%MOD);

        // TEST 7: Inverso modular (inv())
        // Verificar que aM * aM.inv() = 1 (mod MOD)
        // Esto verifica que inv() calcula correctamente el inverso multiplicativo
        assert(aM * aM.inv() == 1);

        // TEST 8: Potencia modular (pow())
        // Verificar que bM.pow(a) = b^a (mod MOD)
        // Comparar con implementación naïve de potencia para validar
        assert(bM.pow(a) == potencia(b, a));
        
    }

    cout << "¡Todos los tests de C++ para Mint pasaron con éxito!" << endl;
    return 0;
}