"""
Tests para Notebook 05: Fundamentos de Machine Learning
========================================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_train_test_split():
    """Test para división train/test"""
    def verificar(X_train, X_test, y_train, y_test):
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total

        if not (0.7 <= train_ratio <= 0.8):
            print_error(f"Ratio incorrecto. Expected: ~0.75, Got: {train_ratio:.2f}")
            return False

        print(f"   ✓ Train set: {len(X_train)} muestras")
        print(f"   ✓ Test set: {len(X_test)} muestras")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_accuracy():
    """Test para cálculo de accuracy"""
    def verificar(accuracy):
        if not (0 <= accuracy <= 1):
            print_error(f"Accuracy fuera de rango [0,1]: {accuracy}")
            return False

        print(f"   ✓ Accuracy: {accuracy:.2%}")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar
