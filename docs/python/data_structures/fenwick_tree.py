# Fenwick Tree

# Un árbol Fenwick funciona de manera similar a un árbol segmentado,
# pero es menos potente, ya que la operación debe ser
# inversible: suma, recuento de frecuencia... funcionan, pero min/-
# max no. La única ventaja real es que es más rápido
# de escribir y ocupa menos espacio (ambos son lineales).

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)

    def update(self, i, delta):
        i += 1
        while i <= self.n:
            self.tree[i] += delta # this is modified to change the operation
            i += i & -i

    def query(self, i):
        i += 1
        res = 0
        while i > 0:
            res += self.tree[i] #this is modified to change the operation
            i -= i & -i
        return res

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1) # This is modified to change the operation