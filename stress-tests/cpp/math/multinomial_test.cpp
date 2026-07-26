#include <bits/stdc++.h>

using namespace std;

#define rep(i, a, b) for (int i = (a); i < (b); ++i)
#define sz(x) (int)(x).size()
typedef long long ll;
typedef vector<int> vi;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/math/multinomial.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    cout << "Iniciando stress test para multinomial (C++)..." << endl;

    {
        vi v = {2, 1};
        assert(multinomial(v) == 3);
    }

    {
        vi v = {1, 1, 1};
        assert(multinomial(v) == 6);
    }

    {
        vi v;
        assert(multinomial(v) == 1);
    }

    mt19937 rd(42);
    for (int t = 1; t <= 50; ++t) {
        int k = 2 + rd() % 4;
        vi v(k);
        for (int i = 0; i < k; ++i) v[i] = rd() % 4;

        ll sum = accumulate(v.begin(), v.end(), 0LL);
        ll expected = 1;
        for (ll i = 2; i <= sum; ++i) expected *= i;
        for (int x : v) {
            ll den = 1;
            for (int i = 2; i <= x; ++i) den *= i;
            expected /= den;
        }
        assert(multinomial(v) == expected);

    }

    cout << "¡Todos los tests de C++ para multinomial pasaron con éxito!" << endl;
    return 0;
}