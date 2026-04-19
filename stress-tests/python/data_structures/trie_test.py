import random
import string
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../docs/python/data_structures')))
from trie import Trie

def random_word(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def run_test():
    trie = Trie()
    ground_truth = set()
    added_words = []

    print("Iniciando stress test para Trie (Python)...")

    # 1. Inserción
    for _ in range(5000):
        w = random_word(random.randint(1, 10))
        trie.insert(w)
        ground_truth.add(w)
        added_words.append(w)

    # 2. Verificación de búsqueda
    for w in added_words:
        assert trie.search(w) == True

    # 3. Verificación de prefijos
    for _ in range(1000):
        prefix = random_word(2)
        expected_has_prefix = any(w.startswith(prefix) for w in ground_truth)
        assert trie.starts_with(prefix) == expected_has_prefix

    print("¡Todos los tests de Python pasaron con éxito!")

if __name__ == "__main__":
    run_test()