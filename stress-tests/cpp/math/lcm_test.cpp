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
    int num_tests = 30; 
    
    for (int t = 1; t <= num_tests; t++) {
        int a = rd() % MAXN + 1;
        int b = rd() % MAXN + 1;

        assert(((long long)a*b)/__gcd(a, b) == lcm(a, b));
        
    }

    cout << "¡Todos los tests de C++ para lcm pasaron con éxito!" << endl;
    return 0;
}