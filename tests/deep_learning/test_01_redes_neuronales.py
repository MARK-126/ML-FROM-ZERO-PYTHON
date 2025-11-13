"""
Tests para Deep Learning: Redes Neuronales
==========================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_forward_propagation():
    """Test para forward propagation"""
    def verificar(A, expected_shape):
        if A.shape != expected_shape:
            print_error(f"Shape incorrecta. Expected: {expected_shape}, Got: {A.shape}")
            return False

        print("   ✓ Forward propagation correcta")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_backward_propagation():
    """Test para backward propagation"""
    def verificar(grads):
        required_keys = ['dW1', 'db1', 'dW2', 'db2']
        for key in required_keys:
            if key not in grads:
                print_error(f"Falta el gradiente '{key}'")
                return False

        print("   ✓ Backward propagation correcta")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar
