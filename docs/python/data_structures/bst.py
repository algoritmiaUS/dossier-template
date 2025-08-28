# Árbol de Búsqueda Binaria (BST)
#Permiten buscar elementos en ellos en tiempo logarítmico. Esencialmente es como realizar una búsqueda binaria
# en una lista ordenada. Sus elementos deben tener un orden parcial.

class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        def _insert(node, key):
            if not node:
                return Node(key)
            if key < node.key:
                node.left = _insert(node.left, key)
            elif key > node.key:
                node.right = _insert(node.right, key)
            return node
        
        self.root = _insert(self.root, key)

    def search(self, key):
        def _search(node, key):
            if not node or node.key == key:
                return node
            if key < node.key:
                return _search(node.left, key)
            else:
                return _search(node.right, key)
        
        return _search(self.root, key)

    def delete(self, key):
        def _min_value_node(node):
            current = node
            while current.left:
                current = current.left
            return current

        def _delete(node, key):
            if not node:
                return None
            
            if key < node.key:
                node.left = _delete(node.left, key)
            elif key > node.key:
                node.right = _delete(node.right, key)
            else:
                # Caso 1: El nodo no tiene hijos o tiene un solo hijo.
                if not node.left:
                    return node.right
                elif not node.right:
                    return node.left
                
                # Caso 2: El nodo tiene dos hijos.
                # Encontramos el sucesor inorden (el nodo más pequeño en el subárbol derecho).
                temp = _min_value_node(node.right)
                
                # Copiamos el contenido del sucesor inorden a este nodo.
                node.key = temp.key
                
                # Eliminamos el sucesor inorden.
                node.right = _delete(node.right, temp.key)
            
            return node
        
        self.root = _delete(self.root, key)