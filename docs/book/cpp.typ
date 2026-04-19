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

#codeblock("../cpp/data_structures/trie.h", "cpp")
#codeblock("../cpp/data_structures/segment_tree.h", "cpp")


= Grafos

Un grafo G=(V,E) es un conjunto de vértices V y aristas (E, que almacena la información de conectividad entre los vértices en V).

#codeblock("../cpp/graphs/Dijkstra.h", "cpp")
#codeblock("../cpp/graphs/dfs.h", "cpp")
#codeblock("../cpp/graphs/bfs.h", "cpp")
#codeblock("../cpp/graphs/BellmanFord.h", "cpp")
#codeblock("../cpp/graphs/FloydWarshall.h", "cpp")
#codeblock("../cpp/graphs/EdmondsKarp.h", "cpp")
#codeblock("../cpp/graphs/Dinic.h", "cpp")
#codeblock("../cpp/graphs/2sat.h", "cpp")

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