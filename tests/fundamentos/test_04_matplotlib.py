"""
Tests para Notebook 04: Matplotlib - Visualización
===================================================
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_plot_basico():
    """Test para gráfico básico"""
    def verificar(fig, ax):
        if fig is None or ax is None:
            print_error("Figura o axes es None")
            return False

        print("   ✓ Gráfico creado correctamente")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_subplots():
    """Test para subplots"""
    def verificar(fig, axes):
        if len(axes) != 2:
            print_error(f"Número incorrecto de subplots. Expected: 2, Got: {len(axes)}")
            return False

        print("   ✓ Subplots creados correctamente")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar
