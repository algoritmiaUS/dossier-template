# Convex Hull (Envolvente Convexa)
#
# El objetivo de este algoritmo es encontrar los puntos de un conjunto
# que forman el perímetro de la figura convexa más pequeña que contiene
# a todos los demás puntos. Piensa en ello como si estuvieras estirando
# una banda elástica alrededor de un conjunto de clavos. Los puntos
# tocados por la banda elástica son la envolvente convexa. 

def convex_hull(points):
    if len(points) <= 1:
        return points

    # Sort by x, then y
    points = sorted(list(set(points)))

    def cross(o, a, b):
        # Cross product of vectors OA and OB
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0: # Add <= here to make it non strict
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:  # Add <= here to make it non strict
            upper.pop()
        upper.append(p)

    # Concatenate lower and upper, removing the last point of each (duplicate)
    return lower[:-1] + upper[:-1]