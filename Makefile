# Configuración de rutas
TYPST_MAIN = docs/book/main.typ
BUILD_DIR = build
STRESS_TESTS_CPP = stress-tests/cpp
STRESS_TESTS_PY = stress-tests/python

# Flags de compilación para C++
CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Wfatal-errors

.PHONY: help pdf-cpp pdf-python test-cpp test-python test clean

help:
	@echo "Dossier CAUS - Gestión Profesional"
	@echo "  make pdf-cpp     : Genera cpp.pdf en el raíz"
	@echo "  make pdf-python  : Genera python.pdf en el raíz"
	@echo "  make test        : Ejecuta todos los tests recursivamente"
	@echo "  make clean       : Borra archivos temporales y PDFs generados"

pdf-cpp:
	typst compile --root . --input language=cpp $(TYPST_MAIN) cpp.pdf
	@echo "Creado: cpp.pdf"

pdf-python:
	typst compile --root . --input language=python $(TYPST_MAIN) python.pdf
	@echo "Creado: python.pdf"

# Stress Tests de C++
test-cpp:
	@echo "Iniciando Stress Tests de C++..."
	@mkdir -p $(BUILD_DIR)/tests
	@find $(STRESS_TESTS_CPP) -name "*.cpp" | while read -r test_file; do \
		rel_path=$${test_file#$(STRESS_TESTS_CPP)/}; \
		test_name=$${rel_path%.cpp}; \
		mkdir -p "$(BUILD_DIR)/tests/$$(dirname "$$rel_path")"; \
		echo "Testing $$rel_path..."; \
		$(CXX) $(CXXFLAGS) "$$test_file" -o "$(BUILD_DIR)/tests/$$test_name" && \
		"./$(BUILD_DIR)/tests/$$test_name" || exit 1; \
	done
	@echo "Tests de C++ finalizados con éxito."

# Stress Tests de Python
test-python:
	@echo "Iniciando Stress Tests de Python..."
	@find $(STRESS_TESTS_PY) -name "*.py" | while read -r test_file; do \
		echo "Testing $$test_file..."; \
		python3 "$$test_file" || exit 1; \
	done
	@echo "Tests de Python finalizados con éxito."

test: test-cpp test-python

clean:
	rm -rf $(BUILD_DIR)
	rm -f cpp.pdf python.pdf
