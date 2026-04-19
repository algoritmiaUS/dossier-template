// Plantilla Python

#import "utils.typ": codeblock
#show math.equation: set text(10pt)

= Preparación concurso 

#codeblock("../python/basic/template.py", "python")
#codeblock("../python/basic/run.sh", "bash")
#codeblock("../python/basic/troubleshoot.txt", "txt")
#codeblock("../python/basic/indefinite_read.py", "python")

= Matemáticas

#codeblock("../python/math/gcd.py", "python")
#codeblock("../python/math/lcm.py", "python")
#codeblock("../python/math/sieve.py", "python")
#codeblock("../python/math/binary-exp.py", "python")
#codeblock("../python/math/binomial.py", "python")

= Estructuras de Datos

#codeblock("../python/data_structures/bst.py", "python")
#codeblock("../python/data_structures/trie.py", "python")
#codeblock("../python/data_structures/segment_tree.py", "python")
#codeblock("../python/data_structures/fenwick_tree.py", "python")

= Grafos

Un grafo G=(V,E) es un conjunto de vértices V y aristas (E, que almacena la información de conectividad entre los vértices en V).

*Lectura de Grafos*: Existen diferentes estructuras de datos para almacenar grafos, no obstante, la más empleada es la lista de Adyacencia, que  abreviaremos como AL. En caso de ver la nomenclatura AM, nos estamos refiriendo a la matriz de adyacencia.
#codeblock("../python/graphs/read_graphs.py", "python")
#codeblock("../python/graphs/dijkstra.py", "python")
#codeblock("../python/graphs/dfs.py", "python")
#codeblock("../python/graphs/bfs.py", "python")
#codeblock("../python/graphs/bellman_ford.py", "python")
#codeblock("../python/graphs/floyd_warshall.py", "python")
#codeblock("../python/graphs/edmonds_karp.py", "python")
#codeblock("../python/graphs/2sat.py", "python")

=  Geometría
#codeblock("../python/geometry/convex_hull.py", "python")

= Apéndices
#codeblock("../python/appendix/techniques.txt", "txt")

