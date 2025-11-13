"""
Tests para Notebook 04: Matplotlib - Visualización de Datos
=============================================================
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_grafico_lineas():
    """Test para crear gráfico de función cuadrática"""
    def verificar(fig):
        if fig is None:
            print_error("La figura es None")
            return False

        if not isinstance(fig, matplotlib.figure.Figure):
            print_error(f"El resultado no es una Figure. Got: {type(fig)}")
            return False

        all_passed = True

        # Obtener los ejes de la figura
        axes = fig.get_axes()
        if len(axes) == 0:
            print_error("La figura no tiene ejes (axes)")
            return False

        ax = axes[0]

        # Verificar que hay líneas graficadas
        lines = ax.get_lines()
        if len(lines) == 0:
            print_error("No se encontraron líneas en el gráfico")
            return False

        # Verificar título
        title = ax.get_title()
        if 'cuadrática' not in title.lower() or 'f(x)' not in title.lower():
            print_error(f"Título incorrecto. Expected: 'Función Cuadrática: f(x) = x² - 4x + 3', Got: '{title}'")
            all_passed = False
        else:
            print("   ✓ Título correcto")

        # Verificar etiquetas de ejes
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel.lower() != 'x':
            print_error(f"Etiqueta X incorrecta. Expected: 'x', Got: '{xlabel}'")
            all_passed = False
        else:
            print("   ✓ Etiqueta X correcta")

        if ylabel.lower() != 'f(x)':
            print_error(f"Etiqueta Y incorrecta. Expected: 'f(x)', Got: '{ylabel}'")
            all_passed = False
        else:
            print("   ✓ Etiqueta Y correcta")

        # Verificar color de línea
        line = lines[0]
        color = line.get_color()
        if color.lower() != 'purple':
            print_error(f"Color de línea incorrecto. Expected: 'purple', Got: '{color}'")
            all_passed = False
        else:
            print("   ✓ Color de línea correcto")

        # Verificar grosor de línea
        linewidth = line.get_linewidth()
        if linewidth != 2:
            print_error(f"Grosor de línea incorrecto. Expected: 2, Got: {linewidth}")
            all_passed = False
        else:
            print("   ✓ Grosor de línea correcto")

        if all_passed:
            print_success("Ejercicio 1 completado!")

        return all_passed

    return verificar


def test_ejercicio_2_histograma():
    """Test para crear histograma"""
    def verificar(fig):
        if fig is None:
            print_error("La figura es None")
            return False

        if not isinstance(fig, matplotlib.figure.Figure):
            print_error(f"El resultado no es una Figure. Got: {type(fig)}")
            return False

        all_passed = True

        # Obtener los ejes
        axes = fig.get_axes()
        if len(axes) == 0:
            print_error("La figura no tiene ejes")
            return False

        ax = axes[0]

        # Verificar que hay patches (barras del histograma)
        patches = ax.patches
        if len(patches) == 0:
            print_error("No se encontraron barras en el histograma")
            return False

        # Verificar número de bins (debería ser 20)
        n_bins = len(patches)
        if n_bins != 20:
            print_error(f"Número de bins incorrecto. Expected: 20, Got: {n_bins}")
            all_passed = False
        else:
            print("   ✓ Número de bins correcto (20)")

        # Verificar título
        title = ax.get_title()
        if 'distribución' not in title.lower() and 'datos' not in title.lower():
            print_error(f"Título incorrecto. Expected: 'Distribución de Datos', Got: '{title}'")
            all_passed = False
        else:
            print("   ✓ Título correcto")

        # Verificar etiquetas
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel.lower() != 'valor':
            print_error(f"Etiqueta X incorrecta. Expected: 'Valor', Got: '{xlabel}'")
            all_passed = False
        else:
            print("   ✓ Etiqueta X correcta")

        if ylabel.lower() != 'frecuencia':
            print_error(f"Etiqueta Y incorrecta. Expected: 'Frecuencia', Got: '{ylabel}'")
            all_passed = False
        else:
            print("   ✓ Etiqueta Y correcta")

        # Verificar color (coral)
        first_patch = patches[0]
        facecolor = first_patch.get_facecolor()
        # coral es aproximadamente (1.0, 0.5, 0.31) en RGB
        # Verificar que tiene componente rojo alto y verde medio
        if facecolor[0] < 0.9 or facecolor[1] > 0.6 or facecolor[1] < 0.4:
            print_error(f"Color incorrecto. Expected: 'coral', Got: {facecolor}")
            all_passed = False
        else:
            print("   ✓ Color correcto (coral)")

        # Verificar alpha
        alpha = first_patch.get_alpha()
        if alpha is None:
            alpha = 1.0
        if abs(alpha - 0.7) > 0.01:
            print_error(f"Alpha incorrecto. Expected: 0.7, Got: {alpha}")
            all_passed = False
        else:
            print("   ✓ Transparencia (alpha) correcta")

        # Verificar borde negro
        edgecolor = first_patch.get_edgecolor()
        if not (edgecolor[0] < 0.1 and edgecolor[1] < 0.1 and edgecolor[2] < 0.1):
            print_error(f"Borde incorrecto. Expected: negro, Got: {edgecolor}")
            all_passed = False
        else:
            print("   ✓ Bordes correctos (negro)")

        if all_passed:
            print_success("Ejercicio 2 completado!")

        return all_passed

    return verificar
