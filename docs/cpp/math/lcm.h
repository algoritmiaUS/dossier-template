ll lcm(int a, int b) {
    return (ll)a / gcd(a, b) * b;  // evitar overflow
}