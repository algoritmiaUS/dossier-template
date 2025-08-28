class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def print_trie(self, node=None, prefix=''):
        '''Solo sirve de ayuda para visualizar el trie'''
        if node is None:
            node = self.root
            
        for char, child in sorted(node.children.items()):
            marker = '*' if child.is_end_of_word else ''
            print('  ' * len(prefix) + f'- {char}{marker}')
            self.print_trie(child, prefix + char)