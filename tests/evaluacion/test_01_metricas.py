"""
Tests para Evaluación: Métricas de Evaluación
=============================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_confusion_matrix():
    """Test para matriz de confusión"""
    def verificar(cm):
        if cm.shape != (2, 2):
            print_error(f"Shape incorrecta. Expected: (2, 2), Got: {cm.shape}")
            return False

        print("   ✓ Matriz de confusión calculada")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_precision_recall():
    """Test para precisión y recall"""
    def verificar(precision, recall):
        if not (0 <= precision <= 1) or not (0 <= recall <= 1):
            print_error("Precision/Recall fuera de rango [0,1]")
            return False

        print(f"   ✓ Precision: {precision:.2%}")
        print(f"   ✓ Recall: {recall:.2%}")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar


def test_ejercicio_3_f1_score():
    """Test para F1-score"""
    def verificar(f1):
        if not (0 <= f1 <= 1):
            print_error(f"F1-score fuera de rango [0,1]: {f1}")
            return False

        print(f"   ✓ F1-score: {f1:.2%}")
        print_success("Ejercicio 3 completado!")
        return True
    return verificar
