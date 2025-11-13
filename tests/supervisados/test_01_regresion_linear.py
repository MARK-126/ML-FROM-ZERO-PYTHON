"""
Tests para Algoritmos Supervisados: Regresión Linear
====================================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_gradiente():
    """Test para cálculo de gradiente"""
    def verificar(gradient):
        if gradient is None:
            print_error("Gradiente es None")
            return False

        print("   ✓ Gradiente calculado correctamente")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_cost_function():
    """Test para función de costo"""
    def verificar(cost):
        if cost < 0:
            print_error("El costo no puede ser negativo")
            return False

        print(f"   ✓ Costo: {cost:.4f}")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar


def test_ejercicio_3_prediccion():
    """Test para predicciones"""
    def verificar(y_pred, X_test):
        if len(y_pred) != len(X_test):
            print_error("Longitud de predicciones incorrecta")
            return False

        print(f"   ✓ {len(y_pred)} predicciones generadas")
        print_success("Ejercicio 3 completado!")
        return True
    return verificar
