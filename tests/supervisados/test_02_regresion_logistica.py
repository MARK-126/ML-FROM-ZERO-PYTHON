"""
Tests para Notebook 02: Regresión Logística
============================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_sigmoid():
    """Test para función sigmoide"""
    def verificar(sigmoid_func):
        if sigmoid_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Sigmoid de 0
        result = sigmoid_func(0)
        expected = 0.5
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"sigmoid(0) incorrecto.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: sigmoid(0) = 0.5")
        
        # Test 2: Sigmoid de valores positivos grandes
        result = sigmoid_func(10)
        expected = 1 / (1 + np.exp(-10))
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"sigmoid(10) incorrecto.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: sigmoid(10) correcto")
        
        # Test 3: Sigmoid de valores negativos grandes
        result = sigmoid_func(-10)
        expected = 1 / (1 + np.exp(10))
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"sigmoid(-10) incorrecto.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: sigmoid(-10) correcto")
        
        # Test 4: Sigmoid de array
        test_array = np.array([-5, 0, 5])
        result = sigmoid_func(test_array)
        expected = 1 / (1 + np.exp(-test_array))
        
        if not np.allclose(result, expected, atol=1e-10):
            print_error(f"sigmoid(array) incorrecto.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: sigmoid funciona con arrays")
        
        # Test 5: Valores extremos (overflow protection)
        result_large = sigmoid_func(1000)
        result_small = sigmoid_func(-1000)
        
        if not (0.99 < result_large <= 1.0):
            print_error(f"sigmoid(1000) debe estar cerca de 1. Got: {result_large}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: manejo de valores extremos positivos")
        
        if not (0.0 <= result_small < 0.01):
            print_error(f"sigmoid(-1000) debe estar cerca de 0. Got: {result_small}")
            all_passed = False
        else:
            print("   ✓ Test 6 pasado: manejo de valores extremos negativos")
        
        # Test 6: Propiedad de simetría: σ(-z) = 1 - σ(z)
        z = 3.5
        sig_z = sigmoid_func(z)
        sig_neg_z = sigmoid_func(-z)
        
        if not np.isclose(sig_neg_z, 1 - sig_z, atol=1e-10):
            print_error(f"No cumple propiedad de simetría: σ(-z) = 1 - σ(z)")
            print(f"   σ({z}) = {sig_z}")
            print(f"   σ({-z}) = {sig_neg_z}")
            print(f"   1 - σ({z}) = {1 - sig_z}")
            all_passed = False
        else:
            print("   ✓ Test 7 pasado: propiedad de simetría")
        
        if all_passed:
            print_success("Ejercicio 1 completado!")
        
        return all_passed
    
    return verificar


def test_ejercicio_2_metricas():
    """Test para cálculo de precision"""
    def verificar(precision_func):
        if precision_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Precision perfecta
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        
        result = precision_func(y_true, y_pred)
        expected = 1.0
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"Precision incorrecta para test 1 (perfecta).")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: precision perfecta")
        
        # Test 2: Caso típico
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
        # TP = 3 (indices 0, 2, 5)
        # FP = 1 (index 6)
        # Precision = 3 / 4 = 0.75
        
        result = precision_func(y_true, y_pred)
        expected = 0.75
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"Precision incorrecta para test 2.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            print("   Hint: Precision = TP / (TP + FP)")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: caso típico")
        
        # Test 3: Sin predicciones positivas
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        
        result = precision_func(y_true, y_pred)
        expected = 0.0
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"Precision incorrecta cuando no hay predicciones positivas.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: sin predicciones positivas")
        
        # Test 4: Todas predicciones falsas
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        # TP = 0, FP = 4
        # Precision = 0 / 4 = 0.0
        
        result = precision_func(y_true, y_pred)
        expected = 0.0
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"Precision incorrecta para test 4 (todas falsas).")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: todas predicciones falsas")
        
        # Test 5: Mitad correctas
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0])
        # TP = 2 (indices 0, 2)
        # FP = 2 (indices 1, 3)
        # Precision = 2 / 4 = 0.5
        
        result = precision_func(y_true, y_pred)
        expected = 0.5
        
        if not np.isclose(result, expected, atol=1e-10):
            print_error(f"Precision incorrecta para test 5.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: mitad correctas")
        
        if all_passed:
            print_success("Ejercicio 2 completado!")
        
        return all_passed
    
    return verificar
