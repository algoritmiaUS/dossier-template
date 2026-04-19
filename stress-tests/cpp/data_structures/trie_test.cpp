#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <cassert>
#include <random>
#include <unordered_map>

// Incluimos tu implementación desde la carpeta de docs
#include "../../../docs/cpp/data_structures/trie.h"

using namespace std;

string random_string(size_t length) {
    static const char charset[] = "abcdefghijklmnopqrstuvwxyz";
    static mt19937 gen(42);
    uniform_int_distribution<> dis(0, sizeof(charset) - 2);
    string s;
    for (size_t i = 0; i < length; ++i) s += charset[dis(gen)];
    return s;
}

int main() {
    Trie trie;
    set<string> words_set;
    vector<string> added_words;

    cout << "Iniciando stress test para Trie (C++)..." << endl;

    // 1. Inserción masiva
    for (int i = 0; i < 5000; ++i) {
    string w = random_string(1 + rand() % 10);
    if (words_set.find(w) == words_set.end()) { // Solo insertar si es nueva
        trie.insert(w);
        words_set.insert(w);
        added_words.push_back(w);
    }
    }
    // 2. Verificar búsqueda de palabras existentes
    for (const string& w : added_words) {
        assert(trie.search(w) == true);
    }

    // 3. Verificar búsqueda de palabras aleatorias
    for (int i = 0; i < 5000; ++i) {
        string w = random_string(5);
        bool expected = (words_set.find(w) != words_set.end());
        assert(trie.search(w) == expected);
    }

    // 4. Verificar prefijos (startsWith)
    for (int i = 0; i < 1000; ++i) {
        string pref = random_string(2);
        bool has_prefix = false;
        int count_expected = 0;
        for (const string& s : words_set) {
            if (s.substr(0, pref.size()) == pref) {
                has_prefix = true;
                count_expected++;
            }
        }
        assert(trie.startsWith(pref) == has_prefix);
        assert(trie.countWordsStartingWith(pref) == count_expected);
    }

    cout << "¡Todos los tests de C++ pasaron con éxito!" << endl;
    return 0;
}