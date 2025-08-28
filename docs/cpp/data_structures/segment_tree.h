// Es una estructura empleada para optimizar
// operaciones sobre rangos (segmentos) de un array.
// Gracias a Dalopir (UCppM)

// Funciones para configurar el segment tree
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

// Configuracion del segment tree
F para las queries, G para las actualizaciones
```cpp
template<typename T> struct STOP {
    using F = Max<T>; using G = Sum<T>;
    // d(g(a, b, ...), x, c): Distribute g over f
    // Ex: max(a+x, b+x, ...) = max(a, b, ...)+x
    // Ex: sum(a+x, b+x, ...) = sum(a, b, ...)+x*c
    static T d(T v, T x, int c) { return G()(v, x); }
};

// Segment Tree Básico
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
