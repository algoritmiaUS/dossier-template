/**
 * Date: 2026-04-19
 * Description: Efectúa operaciones modulo MOD de forma eficiente
 */

const int MOD = 998244353;
struct Mint { // Es una estructura como el int pero que trabaja en mod MOD
    int v;
    Mint(ll val = 0) {
        v = int(val % MOD);
        if (v < 0) v += MOD;
    }
    Mint operator+(const Mint &o) const { return Mint(v + o.v); }
    Mint operator-(const Mint &o) const { return Mint(v - o.v); }
    Mint operator*(const Mint &o) const { return Mint(1LL * v * o.v); }
    Mint operator/(const Mint &o) const { return *this * o.inv(); }
    bool operator<(const Mint& o) const {return v < o.v;}
    bool operator>(const Mint& o) const {return v > o.v;}
    bool operator==(const Mint& o) const {return v == o.v;}
    Mint pow(ll p) const {
        Mint a = *this, res = 1;
        while (p > 0) {
            if (p & 1) res = res * a;
            a = a * a;
            p >>= 1;
        }
        return res;
    }
    Mint inv() const { return pow(MOD - 2); }
    friend ostream& operator<<(ostream& os, const Mint& m) {
        os << m.v; // cout
        return os;
    }
};
istream& operator>>(std::istream& input, Mint& m) {
    input >> m.v; // cin
    return input;
}