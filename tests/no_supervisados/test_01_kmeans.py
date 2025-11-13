"""
Tests para Algoritmos No Supervisados: K-Means
==============================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_inicializar_centroides():
    """Test para inicialización de centroides"""
    def verificar(centroides, k):
        if len(centroides) != k:
            print_error(f"Número de centroides incorrecto. Expected: {k}, Got: {len(centroides)}")
            return False

        print(f"   ✓ {k} centroides inicializados")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_asignar_clusters():
    """Test para asignación de clusters"""
    def verificar(labels, n_samples):
        if len(labels) != n_samples:
            print_error("Longitud de labels incorrecta")
            return False

        print(f"   ✓ {n_samples} puntos asignados a clusters")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar
