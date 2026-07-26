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
    int num_tests = 30; 
    
    for (int t = 1; t <= num_tests; t++) {
        int a = rd() % MAXA + 1;
        int b = rd() % MAXB + 1;
        while(b == a) b = rd() % MAXB + 1;
        if (a%MOD > b%MOD)swap(a, b);

        Mint aM = a, bM = b;
        // 1. Test de igualdad
        assert(aM == a);

        // 2. Test de menor que
        assert(aM < bM);

        // 3. Test de mayor que
        assert(bM > aM);

        // 4. Test de suma
        assert(aM+bM == (a+b)%MOD);

        // 5. Test de resta
        assert(aM-bM == (a-b)%MOD);

        // 6. Test de multiplicación
        assert(aM*bM == ((ll)a*b)%MOD);

        // 7. Test de inverso
        assert(aM * aM.inv() == 1);

        // 8. Test de potencia
        assert(bM.pow(a) == potencia(b, a));
        
    }

    cout << "¡Todos los tests de C++ para Mint pasaron con éxito!" << endl;
    return 0;
}