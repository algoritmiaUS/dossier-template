# Criba de Erastótenes
# Algoritmo para generar números primos
# Sacado de: https://community.lambdatest.com/t/how-can-i-optimize-the-sieve-of-eratosthenes-in-python-for-larger-limits/34557/3

def eratosthene(limit):
    primes = [2, 3, 5]
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False

    return [i for i in range(2, limit + 1) if sieve[i]]