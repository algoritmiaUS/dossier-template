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

Implementaciones de estructuras de datos no estándar que estén implementadas en la librería estándar

#codeblock("../python/data_structures/bst.py", "python")
#codeblock("../python/data_structures/trie.py", "python")
#codeblock("../python/data_structures/segment_tree.py", "python")
#codeblock("../python/data_structures/fenwick_tree.py", "python")

= Grafos

Un grafo G=(V,E) es un conjunto de vértices V y aristas (E, que almacena la información de conectividad entre los vértices en V).

*Lectura de Grafos*: Existen diferentes estructuras de datos para almacenar grafos, no obstante, la más empleada es la lista de Adyacencia, que  abreviaremos como AL. En caso de ver la nomenclatura AM, nos estamos refiriendo a la matriz de adyacencia.
#codeblock("../python/graphs/read_graphs.py", "python")


*Dijkstra:*
Se utiliza para encontrar el camino más corto desde un nodo de inicio hasta todos los demás nodos en un grafo ponderado.

#codeblock("../python/graphs/dijkstra.py", "python")

*DFS:*
Recorre todos los nodos de un grafo o árbol profundizando en cada rama antes de retroceder.
#codeblock("../python/graphs/dfs.py", "python")


*BFS:*
Recorre todos los nodos de un grafo o árbol nivel por nivel.
#codeblock("../python/graphs/bfs.py", "python")



*Bellman-Ford:*
Calcula los caminos más cortos desde $s$ en un grafo que puede tener aristas con pesos negativos.
- Los nodos inalcanzables obtienen `dist = inf`; los nodos alcanzables a través de ciclos de peso negativo obtienen `dist = -inf`.
- Se asume que $V^2 dot.op max|w_i | < 2^63$.

#codeblock("../python/graphs/bellman_ford.py", "python")


*FloydWarshall:*
Calcula las distancias más cortas entre todos los pares de un grafo dirigido que podría tener aristas con pesos negativos.

- La entrada es una matriz de distancias m, donde m[i][j] = inf si i y j no son adyacentes.
- Como salida, m[i][j] se establece en la distancia más corta entre i y j, inf si no existe camino,
  o -inf si el camino pasa por un ciclo de peso negativo.

#codeblock("../python/graphs/floyd_warshall.py", "python")


*EdmondsKarp:*
Algoritmo de flujo con complejidad garantizada $O(V E^2)$.  
Para obtener los valores de flujo de las aristas, compara las capacidades antes y después, y toma solo los valores positivos.


#codeblock("../python/graphs/edmonds_karp.py", "python")




=  Geometría

#codeblock("../python/geometry/convex_hull.py", "python")

= Apéndices
#codeblock("../python/appendix/techniques.txt", "txt")

