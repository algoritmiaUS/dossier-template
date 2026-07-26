#include <bits/stdc++.h>

using namespace std;

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/data_structures/segment_tree.h"

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    mt19937 rd(42);

    cout << "Iniciando stress test para Segment Tree (C++)..." << endl;

    const int n = 64;
    vector<int> a(n);
    for (int i = 0; i < n; ++i) a[i] = int(rd() % 1000) - 500;

    ST<int> st(a);

    for (int t = 1; t <= 200; ++t) {
        int op = rd() % 3;
        if (op == 0) {
            int i = rd() % n;
            int delta = int(rd() % 200) - 100;
            a[i] += delta;
            st.apply(i, delta);
        } else if (op == 1) {
            int i = rd() % n;
            int value = int(rd() % 1000) - 500;
            a[i] = value;
            st.set(i, value);
        } else {
            int l = rd() % n;
            int r = rd() % n;
            if (l > r) swap(l, r);
            int expected = *max_element(a.begin() + l, a.begin() + r + 1);
            assert(st.query(l, r) == expected);
        }

        for (int i = 0; i < n; ++i) {
            assert(st.get(i) == a[i]);
        }
    }

    cout << "¡Todos los tests de C++ para segment tree pasaron con éxito!" << endl;
    return 0;
}