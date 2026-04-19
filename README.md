# 📖 Dossier CAUS  

Este dossier es la plantilla oficial del **Club de Algoritmia de la Universidad de Sevilla** para competiciones de programación en **Python**. En el futuro, planeamos extenderlo a más lenguajes.  

Está escrito en [Typst](https://typst.app/), una alternativa a LaTeX desarrollada en Rust, elegida por su velocidad y eficiencia. Optamos por Typst en lugar de LaTeX debido al alto coste computacional de las plantillas tradicionales.  

## 

```
dossier-template/
├── Makefile                 # Automatiza 'make test' y 'make pdf'
├── stress-tests/            # Directorio de pruebas
│   ├── cpp/
│   │   ├── FenwickTree.cpp
│   │   └── ...
│   └── python/
│       ├── fenwick_test.py
│       └── ...
├── docs/                    # Todo lo que genera el dossier
│   ├── assets/              # Logo e imágenes
│   ├── book/                # Archivos Typst (main, template, utils)
│   ├── cpp/                 # Solo el código C++ limpio (.h)
│   └── python/              # Solo el código Python limpio (.py)
└── .github/                 # CI/CD para compilar y testear
```

## 🚀 Cómo usar la plantilla  

1. Clona este repositorio o descárgalo en formato `.zip` y descomprímelo. 
3. Abre tu editor de texto favorito (en este caso usaremos como ejemplo VS Code)
3. Descarga la extensión [Tinymist](https://github.com/Myriad-Dreamin/tinymist). Está disponible para VS Code y en editores como NVIM, EMACS, Sublime Text, entre otros.
4. Editar tu proyecto, para ver las visualizaciones, en la esquina superior derecha, pulsar el botón `Tyspt Preview`, os abrirá una pestaña con la visualización en tiempo real.
5. Para cambiar tu lenguaje de programación preferido en el fichero `main.typ` solamente cambiar la linea `let value = sys.inputs.at("language", default: "python")`` por el lenguaje por defecto 

![1](assets/local-example.png)

## 📜 Licencia  

Al contribuir a este proyecto, aceptas que tu código se publique bajo los términos de la [Licencia MIT](LICENSE.txt).