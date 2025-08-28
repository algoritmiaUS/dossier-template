// Criba de Erastótenes: Algoritmo para generar
// números primos. Gracias a noahdris (UCppM)
int N = 30;
vector<bool> es_primo(N+1,true);
vector<int> primos;
for(int i = 2; i <= N; i++){
    if(es_primo[i]){
        primos.push_back(i);
        for(int j = i; j*i <= N; j++) es_primo[j*i] = false;
    }
}