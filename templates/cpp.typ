= Setup
// Plantilla CPP
// Contribuidores:
// Kenny Jesús Flores Huamán,
// Pablo Dávila Herrero
// Dalopir (UCppM)
// noahdris (UCppM)
// Nacho Los BoquerO(n³) UNED

Cuando trabajas en la terminal, usa este comando para compilar y ejecutar tu programa con entradas desde un archivo (ej: input.txt)

```cpp
g++ main.cpp -o main && ./main < input.txt
```

== Librería estándar

Es un header no estándar que incluye todas las bibliotecas estándar de C++ de una sola vez.

```cpp
#include <bits/stdc++.h>
```

== Lectura línea a línea

```cpp
std::string line;
while(std::getline(std::cin, line)){ ... }

```

== Lectura número a número

```cpp
int n;
while(std::cin >> n){ ... }
```


== Lectura carácter a carácter

No ignora espacios ni saltos de línea
```
char c;
while(std::cin::get(c)){ ... }
```

== Optimización de E/S

En problemas con entradas/salidas grandes (ej.: programación competitiva), estas líneas aceleran `cin/cout`:

```cpp
int main() {
    // Desincroniza C++ de C (¡cin/cout más rápidos!)
    std::ios_base::sync_with_stdio(false);
    // Evita que cin vacíe cout antes de cada lectura
    std::cin.tie(nullptr);                  
    // tu código...
}
```

= Operaciones con cadenas

== split

```cpp
std::vector<std::string> split(const std::string & str, char delimiter){
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while(std::getline(ss, token, delimiter)){
        if(!token.empty()) {tokens.push_back(token); }
    }
    return tokens;
}
```
== concatenación

```cpp
std::stringstream msg;
msg << "hola";
std::cout << msg.str() << "\n";
```

== Búsqueda subcadena
```cpp
std::string str = "Hello World!";
if(str.find("World") != std::string::npos) { /* Exists*/}
```

==  Parseo entero

```cpp
int num =  std::stoi(str);
long num = std::stol(str);
long long num = std::stoll(str);
```

== Parseo flotante
```cpp
float num = std::stof(str);
double num = std::stod(str);
long double num = std::stold(str)

```


== Parseo múltiples números
```cpp
std::istringstream iss("123 456");
int num1, num2;
iss >> num1 >> num2; // num1 = 123, num2=456
```

= Estructuras de Datos


// == Listas

// == Pilas (Stacks)


// == Colas (Queues)



// == Sets


// == Diccionarios



// == Heap (Colas de prioridad)


// == Cola doblemente enlazada (Deque)




== Segment Tree

Es una estructura empleada para optimizar operaciones sobre rangos (segmentos) de un array.
Gracias a Dalopir (UCppM)

=== Funciones para configurar el segment tree

```cpp
template<typename T> struct Min {
    T neutral = INT_MAX;
    T operator()(T x, T y) { return min(x, y); }
    T rep(T x, int c) { return x; }
};

template<typename T> struct Max {
    T neutral = INT_MIN;
    T operator()(T x, T y) { return max(x, y); }
    T rep(T x, int c) { return x; }
};

template<typename T> struct Sum {
    T neutral = 0;
    T operator()(T x, T y) { return x+y; }
    T inv(T x) { return -x; }
    T rep(T x, int c) { return x*c; }
};

template<typename T> struct Mul {
    T neutral = 1;
    T operator()(T x, T y) { return x*y; }
    T inv(T x) { return 1/x; }
    T rep(T x, int c) { return pow(x, c); }
};
```


=== Configuracion del segment tree
F para las queries, G para las actualizaciones
```cpp
template<typename T> struct STOP {
    using F = Max<T>; using G = Sum<T>;
    // d(g(a, b, ...), x, c): Distribute g over f
    // Ex: max(a+x, b+x, ...) = max(a, b, ...)+x
    // Ex: sum(a+x, b+x, ...) = sum(a, b, ...)+x*c
    static T d(T v, T x, int c) { return G()(v, x); }
};
```
=== Segment Tree Básico

```cpp
template<typename T, typename OP = STOP<T>> struct ST {
    typename OP::F f; typename OP::G g;
    ST *L = 0, *R = 0; int l, r, m; T v;

    ST(const vector<T> &a, int ql, int qr)
    : l(ql), r(qr), m((l+r)/2) {
        if (ql == qr) v = a[ql];
        else L = new ST(a, ql, m), R = new ST(a, m+1, qr),
             v = f(L->v, R->v);
    }
    ST(const vector<T> &a) : ST(a, 0, a.size()-1) {}
    ~ST() { delete L; delete R; }

    T query(int ql, int qr) {
        if (ql <= l && r <= qr) return v;
        if (r < ql || qr < l) return f.neutral;
        return f(R->query(ql, qr), L->query(ql, qr));
    }
    void apply(int i, T x) {
        if (l == r) { v = g(x, v); return; }
        if (i <= m) L->apply(i, x);
        else        R->apply(i, x);
        v = f(L->v, R->v);
    }
    void set(int i, T x) {
        if (l == r) { v = x; return; }
        if (i <= m) L->set(i, x);
        else        R->set(i, x);
        v = f(L->v, R->v);
    }

    T get(int i) { return query(i, i); }
};
```



= Algoritmos de Búsqueda y ordenamiento


== Búsqueda binaria

Busca en una lista ordenada dividiéndola a la mitad en cada paso. La complejidad en tiempo es de O(log n).

=== Comprobar si existe elemento

```cpp
bool exists = binary_search(vec.begin(), vec.end(), value);
```

=== Posición del elemento

```cpp
auto it = lower_bound(vec.begin(), vec.end(), value);
if (it != vec.end() && *it == value) {
  int index = distance(vec.begin(), it); // posición del elemento
  cout << "Posición exacta: " << index << '\n';

}
```

=== Elemento más pequeño que verifica ser más grande que value
```cpp
it = upper_bound(vec.begin(), vec.end(), value);
if (it != vec.end()) {
  int index = distance(vec.begin(), it);
  cout << "Más pequeño que es mayor a value: " << index << '\n';
}
```

=== Elemento más pequeño que verifica ser mayor o igual a "value"
```cpp
it = lower_bound(vec.begin(), vec.end(), value);
  if (it != vec.end()) {
    int index = distance(vec.begin(), it);
     cout << "Más pequeño que es mayor o igual a value: " << index << '\n';
    }
```

=== Elemento más grande menor que "value"
```cpp
it = lower_bound(vec.begin(), vec.end(), value);
  if (it != vec.begin()) {
    --it; // retrocede al elemento anterior
    int index = distance(vec.begin(), it);
    cout << "Más grande menor que value: " << index << '\n';
}
```

=== Elemento más grande menor o igual que "value"
```cpp
it = upper_bound(vec.begin(), vec.end(), value);
if (it != vec.begin()) {
  --it; // retrocede al elemento anterior
  if (*it <= value) {
    int index = distance(vec.begin(), it);
    cout << "Más grande menor o igual que value: " << index << '\n';
  }
}
```


= Grafos

Un grafo G=(V,E) es sencillamente, un conjunto de vértices V y aristas (E, que almacena la información de conectividad entre los vértices en V).

// == Lectura de Grafos

// Existen diferentes estructuras de datos para almacenar grafos, no obstante, la más empleada es la lista de Adyacencia, que  abreviaremos como AL. En caso de ver la nomenclatura AM, nos estamos refiriendo a la matriz de adyacencia.


== Algoritmo de Dijkstra

Se utiliza para encontrar el camino más corto desde un nodo de inicio hasta todos los demás nodos en un grafo ponderado, con el objetivo de resolver problemas como rutas más eficientes en redes, mapas de carreteras o en la optimización de costos de conexión.

```cpp
int main() {
    int V, E;
    cin >> V >> E;

    vector<vector<pair<int, int>>> edges(V);
    for (int i = 0; i < E; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        edges[u].push_back({v, w});
        edges[v].push_back({u, w});
    }

    vector<int> dist(V, INT_MAX);
    vector<bool> used(V, false);
    priority_queue<pair<int, int>> pq;

    dist[0] = 0, pq.push({0, 0});
    while (!pq.empty()) {
        auto [uw, u] = pq.top(); pq.pop(); uw *= -1;
        if (used[u]) continue; used[u] = true;

        for (auto [v, w] : edges[u]) {
            int vw = uw+w;
            if (vw < dist[v])
                dist[v] = vw, pq.push({-vw, v});
        }
    }

    cout << dist[V-1] << "\n";
}
```


== Algoritmo de Floyd-Warshall

Es útil cuando necesitas calcular las distancias más cortas entre todos los pares de vértices en un grafo. En general no es buena idea utilizarlo cuando $450 < |V|$.

```cpp
//TODO
```


== Búsqueda en Profundidad (DFS)
```cpp

```

== Búsqueda en Amplitud (BFS)

Recorre todos los nodos de un grafo o árbol nivel por nivel.

```cpp
// TODO
```


// == Flujo máximo

// TODO Dar pistas de cómo utilizar esto. El formato de entrada de las funciones, etc.

```cpp
//TODO:
```


= Matemáticas


== Máximo común divisor (GCD)

```cpp
int gcd(int a, int b) {
    while (b) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}
```


== Mínimo común múltiplo (LCM)

```cpp
int lcm(int a, int b) {
    return a / gcd(a, b) * b;  // evitar overflow
}
```


== Criba de Erastótenes

Algoritmo para generar números primos


```cpp
// Proporcionado por noahdris (UCppM)
int N = 30;
vector<bool> es_primo(N+1,true);
vector<int> primos;
for(int i = 2; i <= N; i++){
    if(es_primo[i]){
        primos.push_back(i);
        for(int j = i; j*i <= N; j++) es_primo[j*i] = false;
    }
}
```


== Exponenciación rápida

// https://cp-algorithms.com/algebra/binary-exp.html
Calcula $a^b mod m$ de manera eficiente.

```cpp
long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}
```


== Coeficientes binomiales


El coeficiente binomial $mat(n ; k)$ representa el número de formas de elegir un subconjunto de k elementos de un conjunto de $n$ elementos. Por ejemplo $mat(4 ; 2) = 6$ porque el conjunto ${1,2,3,4}$ tiene 6 subconjuntos de 2 elementos ${1,2},{1,3},{1,4},{2,3},{2,4},{3,4}$. Para calcular estos coeficientes se utiliza la fórmula $mat(n ; k) = n! / (k!(n-k)!)$.

```cpp
// TODO
```


= Strings


== Trie


```cpp
// Trie construido en base al código de Los BoquerO(n³) UNED
class TrieNode {
public:
    std::unordered_map<char, TrieNode*> children;
    bool isEndOfWord;
    int count;
    
    TrieNode() : isEndOfWord(false), count(0) {}

    ~TrieNode() {
        for(auto& pair : children) {
            delete pair.second;
        }
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    void insert(std::string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
            node->count++;
        }
        node->isEndOfWord = true;
    }

    bool search(std::string word) {
        TrieNode* node = root;
        for (char c : word) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return node != nullptr && node->isEndOfWord;
    }

    bool startsWith(std::string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            if (node->children.find(c) == node->children.end()) {
                return false;
            }
            node = node->children[c];
        }
        return true;
    }

    int countWordsStartingWith(std::string prefix) {
        TrieNode* node = root;
        for (char c : prefix) {
            if (node->children.find(c) == node->children.end()) {
                return 0; // Prefix not found
            }
            node = node->children[c];
        }
        return node->count;
    }
};
```

// == Búsqueda de patrones

// == Hashing de strings

// Para comparar strings eficientemente.


=  Geometría

== Clase Punto

```cpp
// Proporcionado por (UCppM)
struct pt{
  uds x,y;
  pt(){}
  pt(uds x, uds y): x(x), y(y){}

  pt operator+(const pt &p) const {return {x + p.x, y + p.y};}
  pt operator*(uds k) const {return {k*x, k*y};}
  pt operator-() const { return *this *-1; }
  pt operator-(const pt &p) const {return *this + (-p);}
  pt operator/(uds d) const { 
    if (eq(d, 0)) throw runtime_error("operator/ Recuerdos de vietADAMan (p/0)"); 
    return *this*(1/d); 
  }

	pt& operator+=(const pt& p) { return *this = *this+p; };
	pt& operator-=(const pt& p) { return *this = *this-p; };
	pt& operator*=(uds k) { return *this = *this*k; };
	pt& operator/=(uds k) { return *this = *this/k; };

  bool operator==(const pt &a) const { return eq(a.x, x) && eq(a.y,y); }
	bool operator!=(const pt &a) const { return !(a == *this); }
	bool operator>(const pt &a) const { return (eq(a.x, x) ? lt(a.y, y): lt(a.x, x));}
	bool operator<(const pt &a) const { return a > *this; }
	pt rot() const {return pt(-y, x);} // rotate 90 degrees counter-clockwise
};
ostream& operator<<(ostream &out, const pt &p) { return out<<p.x<<" "<<p.y;}
pt operator*(uds k, const pt &p) { return p*k; }
uds dot(const pt &p, const pt &q) { return p.x*q.x + p.y*q.y; }
uds norma(const pt &p) { return sqrt(dot(p,p)); }
uds angle(const pt &p, const pt &q) {return acos(dot(p,q)/(norma(p)*norma(q)));}
uds cross(const pt &p, const pt &q) {return p.x*q.y - p.y*q.x;}

bool collinear(pt a, pt b) { return eq(cross(a, b), 0); } //si son li 

uds orient(pt a, pt b, pt c) {return cross(b-a, c-a);} // 
int orientS(pt a, pt b, pt c) {return sgn(orient(a, b, c)); }

bool cw(pt a, pt b, pt c, bool in_colli) {
  int o = orientS(a, b, c); return o<0 || (in_colli && o==0); // o < 0
}
bool ccw(pt a, pt b, pt c, bool in_colli) {
  int o = orientS(a, b, c); return o>0 || (in_colli && o==0); // o > 0
}
bool collinear(pt a, pt b, pt c) { return eq(orient(a, b, c),0); } // ver si tres puntos estan alineados
```

== Clase Linea

```cpp
struct line{
  pt p, v;
  line(pt v): p(pt(0,0)), v(v){}
  line(pt p, pt v): p(p), v(v){}
	line(uds a, uds b, uds c): p((eq(b,0) ? pt(-c/a, 0) : pt(0, -c/b) )), v(pt(b,-a)){} // ax+by+c=0

  bool in_line(const pt &q) const {return collinear(p-q , v);}
  bool operator ==(const line &r) const {return in_line(r.p) && collinear(v, r.v);}
  bool operator !=(const line &r) const {return !(*this == r);}

  line perp(const pt &b) const {return {b, v.rot()};}
};

vector<pt> intersection(line l1, line l2) {
	if(l1 == l2) throw runtime_error("intersection() son la misma");
	uds d = cross(l1.v, l2.v), c1 = cross(l1.v,l1.p), c2 = cross(l2.v,l2.p);
	if (eq(d,0)) return {};
	return {(l2.v*c1-l1.v*c2)/d};
}

pt proj(const pt &p, const line &r){return intersection(r, r.perp(p))[0];}
pt proj(const line &r, const pt &p){return proj(p, r);}

uds dist(const pt &p, const pt &q){return norma(p-q);}
uds dist(const pt &p, const line &r){return dist(p, proj(p,r));}
uds dist(const line &r, const pt &p){return dist(p,r);}
uds dist(const line &r, const line &s){return (collinear(r.v,s.v) ? dist(r,s.p): 0);}
```


== Clase Polígono

```cpp
struct poly {
	vector<pt> pts;
	poly(){}
	poly(vector<pt> pts): pts(pts){}

	uds perim() const {
    uds result = 0;
    for(int i = 0; i < pts.size(); i++) result += norma(pts[i] - pts[ (i+1)% pts.size()]);
    return result;
	}

	bool isConvex() const {
		bool hasPos=false, hasNeg=false;
		for (int i=0, n=pts.size(); i<n; i++) {
			int o = orientS(pts[i], pts[(i+1)%n], pts[(i+2)%n]);
			if (o > 0) hasPos = true;
			if (o < 0) hasNeg = true;
		}
		return !(hasPos && hasNeg);
	}

	uds area() const{
		uds result = 0;
		for(int i = 0; i<pts.size(); i++) result += cross(pts[i] , pts[ (i+1)% pts.size()]);
		return abs(result)/2;
	}
    //   Check if point belongs to the polygon in  n  //
	bool inPol(const pt &point) const { // linear
		if(pts.size() == 0) return false;
		uds sum = 0;
		for(int i = 0; i<pts.size(); i++){
			if(point == pts[i] || point == pts[(i+1)%pts.size()]) return true;
			sum += angle(pts[i]-point, pts[(i+1)%pts.size()]-point)*(2*ccw(point, pts[i], pts[(i+1)%pts.size()],false) -1);
		}
		return eq(abs(sum), 2*pi);
	}

	bool inConvex(pt &point, bool strict = true) const { // log
		int a = 1, b = pts.size() - 1, c;
    if (orientS(pts[0], pts[a], pts[b]) > 0) swap(a, b);
    if (orientS(pts[0], pts[a], point) > 0 || orientS(pts[0], pts[b], point) < 0 ||
			 (strict && (orientS(pts[0], pts[a], point) == 0 || orientS(pts[0], pts[b], point) == 0)))
			return false;
    while (abs(a - b) > 1) {
			c = (a + b) / 2;
			if (orientS(pts[0], pts[c], point) > 0) b = c;
			else a = c;
    }
    return orientS(pts[a], pts[b], point) < 0 ? true : !strict;
	}

	void convex_hull(bool in_colli = false) {
		pt p0 = *min_element(pts.begin(), pts.end());
		sort(pts.begin(), pts.end(), [&p0](const pt& a, const pt& b) {
			int o = orientS(p0, a, b); return o<0 || (o==0 && lt(dist(p0,a), dist(p0,b)));
		});
		if (in_colli) {
			int i = pts.size()-1;
			for(;i && collinear(p0, pts[i], pts.back()); i--);
			reverse(pts.begin()+i+1, pts.end());
		}
		vector<pt> st;
		for (int i = 0; i < pts.size(); i++) {
			while (st.size() > 1 && !cw(st[st.size()-2], st.back(), pts[i], in_colli)) st.pop_back();
			st.push_back(pts[i]);
		}
		if (!in_colli && st.size() == 2 && st[0] == st[1]) st.pop_back();
		pts = st;
	}

};
uds pick(int I,int B){return I-1 + B*0.5;} // Teorema de Pick. I interior y B borde
```