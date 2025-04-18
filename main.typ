#import "template.typ": dossier 

#show: dossier.with(
  university: "Universidad de Sevilla",
  team_name: "Teorema del Sándwich de Ham",
  members: (
    "Kenny Flores",
    "Pablo Dávila",
    "Pablo Reina"
  ),
  date: datetime.today(),
  num_cols:3
)

// https://forum.typst.app/t/can-i-configure-my-document-e-g-draft-release-version-color-theme-when-creating-a-pdf-without-modifying-the-typst-file-directly/160
#let language = {
  let valid-values = ("python", "cpp", "java")
  let value = sys.inputs.at("language", default: "python") // CAMBIAR LENGUAJE AQUÍ
                                                           // PARA VISUALIZACIÓN
  assert(value in valid-values, message: "`--input language` must be in {valid-values}")
  value
}

#include "templates/"+language+".typ"


= Técnicas

== Recursión

=== Divide y vencerás
- Encontrar puntos interesantes en N log N
=== Algoritmo codicioso (Greedy)
- Planificación
- Máxima suma de subvector contiguo
- Invariantes
- Codificación Huffman

== Teoría de grafos
- Grafos dinámicos (registro adicional)
- Búsqueda en amplitud
- Búsqueda en profundidad
    - Árboles normales / Árboles DFS
- Algoritmo de Dijkstra
- MST: Algoritmo de Prim
- Bellman-Ford
- Teorema de Konig y cobertura de vértices
- Flujo máximo de costo mínimo
- Conmutador de Lovasz
- Teorema del árbol de matriz
- Emparejamiento máximo, grafos generales
- Hopcroft-Karp
- Teorema del matrimonio de Hall
- Secuencias gráficas
- Floyd-Warshall
- Ciclos de Euler
- Flujo máximo
    - Caminos aumentantes
    - Edmonds-Karp
- Emparejamiento bipartito
- Cobertura mínima de caminos
- Ordenación topológica
- Componentes fuertemente conectados
- 2-SAT
- Vértices de corte, aristas de corte y componentes biconectados
- Coloreado de aristas
    - Árboles
- Coloreado de vértices
    - Grafos bipartitos (=> árboles)
    - 3^n (caso especial de cobertura de conjuntos)
- Diámetro y centroide
- K-ésimo camino más corto
- Ciclo más corto

== Programación dinámica
- Mochila
- Cambio de monedas
- Subsecuencia común más larga
- Subsecuencia creciente más larga
- Número de caminos en un DAG
- Camino más corto en un DAG
- Programación dinámica sobre intervalos
- Programación dinámica sobre subconjuntos
- Programación dinámica sobre probabilidades
- Programación dinámica sobre árboles
- Cobertura de conjunto 3^n
- Divide y vencerás
- Optimización de Knuth
- Optimización del cascarón convexo
- RMQ (tabla dispersa, también conocido como saltos 2^k)
- Ciclo bitónico
- Partición logarítmica (bucle sobre lo más restringido)


== Combinatoria
- Cálculo de coeficientes binomiales
// - Principio del casillero
// - Inclusión/exclusión
// - Número de Catalan
// - Teorema de Pick

== Teoría de números
- Partes enteras
- Divisibilidad
- Algoritmo de Euclides
- Aritmética modular
    - Multiplicación modular
    - Inversos modulares
    - Exponentiación modular por cuadrados
- Teorema del resto chino
// - Teorema pequeño de Fermat
- Teorema de Euler
- Función Phi
// - Número de Frobenius
// - Reciprocidad cuadrática
// - Pollard-Rho
// - Miller-Rabin
// - Elevación de Hensel
// - Salto de raíces de Vieta

// == Teoría de juegos
// - Juegos combinatorios
// - Árboles de juego
// - Mini-máximo
// - Nim
// - Juegos en grafos
// - Juegos en grafos con bucles
// - Números de Grundy
// - Juegos bipartitos sin repetición
// - Juegos generales sin repetición
// - Poda alfa-beta

// == Teoría de probabilidades

== Optimización
- Búsqueda binaria
- Búsqueda ternaria
// - Unimodalidad y funciones convexas
// - Búsqueda binaria sobre derivadas

// == Métodos numéricos
// - Integración numérica
// - Método de Newton
// - Búsqueda de raíces con búsqueda binaria/ternaria
// - Búsqueda de sección dorada

// == Matrices
// - Eliminación de Gauss
// - Exponenciación por duplicación

// == Ordenación
// - Ordenación Radix

== Geometría
- Coordenadas y vectores
  * Producto cruzado
  * Producto escalar
- Envolvente convexa
- Corte de polígonos
- Par más cercano
- Compresión de coordenadas
// - Quadtrees
// - KD-trees
- Intersección de todos los segmentos
// - Barrido
//   - Discretización (convertir a eventos y barrer)
//   - Barrido de ángulos
//   - Barrido de líneas
//   - Derivadas segundas discretas

== Strings
- Subcadena común más larga
- Subsecuencias palindrómicas
// - Knuth-Morris-Pratt
- Tries
// - Hashes polinomiales rodantes
// - Arreglo de sufijos
// - Árbol de sufijos
// - Aho-Corasick
- Algoritmo de Manacher
- Listas de posiciones de letras

== Búsqueda combinatoria
- Encuentro en el medio
- Fuerza bruta con poda
- Mejor primero (A estrella)
- Búsqueda bidireccional
- Profundización iterativa DFS / A estrella

// == Estructuras de datos
// - LCA (saltos de 2^k en árboles en general)
// - Técnica de tirar/empujar en árboles
// - Descomposición heavy-light
// - Descomposición en centroides
// - Propagación perezosa
// - Árboles autoequilibrantes
// - Truco de envolvente convexa (wcipeg.com/wiki/Convex_hull_trick)
// - Colas monótonas / pilas monótonas / colas deslizantes
// - Cola deslizante usando 2 pilas
// - Árbol de segmentos persistente
