/**
 * Date: 2026-07-09
 * Description: Calcula los números primos menores o iguales que n
 * Time: O(n log log n)
 */
#include <bits/stdc++.h>
vector<bool> es_primo;
vector<int> primos;
void criba(int N) {
    es_primo.assign(N+1, true);
    primos.clear();
    for(int i = 2; i <= N; i++){
        if(es_primo[i]){
            primos.push_back(i);
            for(int j = i; (long long) j*i <= N; j++) es_primo[j*i] = false;
        }
    }
}