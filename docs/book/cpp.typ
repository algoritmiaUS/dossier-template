// Plantilla CPP

#import "utils.typ": codeblock
#show math.equation: set text(10pt)

= Preparación concurso 

#codeblock("../cpp/basic/template.cpp", "cpp")
#codeblock("../cpp/basic/run.sh", "bash")
#codeblock("../cpp/basic/troubleshoot.txt", "txt")


= Matemáticas

#codeblock("../cpp/math/gcd.h", "cpp")
#codeblock("../cpp/math/lcm.h", "cpp")
#codeblock("../cpp/math/sieve.h", "cpp")
#codeblock("../cpp/math/binary-exp.h", "cpp")

*Coeficientes Multinomiales*

EL siguiente código calcula:
$ binom(k_1 + \u{2026} + k_n, k_1, k_2, \u{2026}, k_n) = (sum_ k_i)/ (k_1 ! k_2 ! \u{2026}  k_n !)$
#codeblock("../cpp/math/multinomial.h", "cpp")

= Estructuras de Datos

Implementaciones de estructuras de datos no estándar de la STL.

#codeblock("../cpp/data_structures/trie.h", "cpp")
#codeblock("../cpp/data_structures/segment_tree.h", "cpp")


= Grafos

Un grafo G=(V,E) es un conjunto de vértices V y aristas (E, que almacena la información de conectividad entre los vértices en V).

*Dijkstra:*
Se utiliza para encontrar el camino más corto desde un nodo de inicio hasta todos los demás nodos en un grafo ponderado.

#codeblock("../cpp/graphs/Dijkstra.h", "cpp")

*DFS:*
Recorre todos los nodos de un grafo o árbol profundizando en cada rama antes de retroceder.
#codeblock("../cpp/graphs/dfs.h", "cpp")


*BFS:*
Recorre todos los nodos de un grafo o árbol nivel por nivel.
#codeblock("../cpp/graphs/bfs.h", "cpp")



*Bellman-Ford:*
Calcula los caminos más cortos desde $s$ en un grafo que puede tener aristas con pesos negativos.
- Los nodos inalcanzables obtienen `dist = inf`; los nodos alcanzables a través de ciclos de peso negativo obtienen `dist = -inf`.
- Se asume que $V^2 dot.op max|w_i | < 2^63$.

#codeblock("../cpp/graphs/BellmanFord.h", "cpp")


*FloydWarshall:*
Calcula las distancias más cortas entre todos los pares de un grafo dirigido que podría tener aristas con pesos negativos.

- La entrada es una matriz de distancias m, donde m[i][j] = inf si i y j no son adyacentes.
- Como salida, m[i][j] se establece en la distancia más corta entre i y j, inf si no existe camino,
  o -inf si el camino pasa por un ciclo de peso negativo.

#codeblock("../cpp/graphs/FloydWarshall.h", "cpp")


*EdmondsKarp:*
Algoritmo de flujo con complejidad garantizada $O(V E^2)$.  
Para obtener los valores de flujo de las aristas, compara las capacidades antes y después, y toma solo los valores positivos.


#codeblock("../cpp/graphs/EdmondsKarp.h", "cpp")


=  Geometría

#codeblock("../cpp/geometry/point.h", "cpp")

*Envolvente Convexa (Convex Hull)*

#grid(
  columns: 2,
  gutter: 5pt,
  "Devuelve un vector con los puntos del envolvente convexo en orden antihorario. Los puntos que se encuentran en el borde de la envolvente entre otros dos puntos no se consideran parte de la envolvente",
  image("../cpp/geometry/convex_hull.jpg")
)
#codeblock("../cpp/geometry/ConvexHull.h", "cpp")


= Apéndices
#codeblock("../cpp/appendix/techniques.txt", "txt")