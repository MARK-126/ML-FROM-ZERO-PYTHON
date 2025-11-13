"""
Tests para Algoritmos Supervisados: K-Nearest Neighbors
=======================================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_distancia_euclidiana():
    """Test para cálculo de distancia euclidiana"""
    def verificar(func):
        # Test con puntos conocidos
        p1 = np.array([0, 0])
        p2 = np.array([3, 4])
        dist = func(p1, p2)

        if not np.isclose(dist, 5.0):
            print_error(f"Distancia incorrecta. Expected: 5.0, Got: {dist}")
            return False

        print("   ✓ Distancia euclidiana calculada correctamente")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_k_vecinos():
    """Test para encontrar k vecinos más cercanos"""
    def verificar(indices, k):
        if len(indices) != k:
            print_error(f"Número de vecinos incorrecto. Expected: {k}, Got: {len(indices)}")
            return False

        print(f"   ✓ {k} vecinos más cercanos encontrados")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar
