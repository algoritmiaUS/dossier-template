/**
 * Author: noahdris (UCppM)
 * Date: 2026-04-19
 * Description: Calcula el GCD de dos números.
 * Time: O(log n)
 */
int N = 30;
vector<bool> es_primo(N+1,true);
vector<int> primos;
for(int i = 2; i <= N; i++){
    if(es_primo[i]){
        primos.push_back(i);
        for(int j = i; j*i <= N; j++) es_primo[j*i] = false;
    }
}