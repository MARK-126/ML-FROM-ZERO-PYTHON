"""
Tests para Algoritmos Supervisados: Regresión Logística
=======================================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_sigmoid():
    """Test para función sigmoid"""
    def verificar(func):
        test_vals = [0, 1, -1, 10, -10]
        for val in test_vals:
            result = func(val)
            if not (0 <= result <= 1):
                print_error(f"Sigmoid({val}) = {result} fuera de rango [0,1]")
                return False

        print("   ✓ Función sigmoid correcta")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_binary_classification():
    """Test para clasificación binaria"""
    def verificar(y_pred):
        unique = np.unique(y_pred)
        if not np.array_equal(unique, [0, 1]) and not np.array_equal(unique, [0]) and not np.array_equal(unique, [1]):
            print_error(f"Predicciones deben ser 0 o 1. Got: {unique}")
            return False

        print("   ✓ Clasificación binaria correcta")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar
