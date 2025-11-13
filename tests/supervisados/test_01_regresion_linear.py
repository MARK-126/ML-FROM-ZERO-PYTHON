"""
Tests para Notebook 01: Regresión Lineal
=========================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_predict():
    """Test para función de predicción lineal"""
    def verificar(predict_func):
        if predict_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Predicción simple
        X1 = np.array([[1, 2], [3, 4], [5, 6]])
        w1 = np.array([0.5, 0.3])
        b1 = 1.0
        
        result1 = predict_func(X1, w1, b1)
        
        if result1 is None:
            print_error("La función retorna None")
            return False
        
        expected1 = np.array([2.1, 3.7, 5.3])
        
        if not np.allclose(result1, expected1, atol=1e-10):
            print_error(f"Predicción incorrecta para test 1.")
            print(f"   Expected: {expected1}")
            print(f"   Got: {result1}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: predicción simple correcta")
        
        # Test 2: Una sola feature
        X2 = np.array([[1], [2], [3], [4]])
        w2 = np.array([2.0])
        b2 = 3.0
        
        result2 = predict_func(X2, w2, b2)
        expected2 = np.array([5.0, 7.0, 9.0, 11.0])
        
        if not np.allclose(result2, expected2, atol=1e-10):
            print_error(f"Predicción incorrecta para test 2 (una feature).")
            print(f"   Expected: {expected2}")
            print(f"   Got: {result2}")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: una feature")
        
        # Test 3: Múltiples features
        X3 = np.array([[1, 2, 3], [4, 5, 6]])
        w3 = np.array([0.1, 0.2, 0.3])
        b3 = 0.5
        
        result3 = predict_func(X3, w3, b3)
        expected3 = np.array([1.4, 3.2])
        
        if not np.allclose(result3, expected3, atol=1e-10):
            print_error(f"Predicción incorrecta para test 3 (múltiples features).")
            print(f"   Expected: {expected3}")
            print(f"   Got: {result3}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: múltiples features")
        
        # Test 4: Verificar que usa operaciones vectorizadas
        X4 = np.array([[1, 2], [3, 4]])
        w4 = np.array([1, 1])
        b4 = 0
        
        result4 = predict_func(X4, w4, b4)
        
        # Verificar tipo de retorno
        if not isinstance(result4, np.ndarray):
            print_error("El resultado debe ser un numpy array")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: retorna numpy array")
        
        # Verificar shape
        if result4.shape != (2,):
            print_error(f"Shape incorrecta. Expected: (2,), Got: {result4.shape}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: shape correcta")
        
        if all_passed:
            print_success("Ejercicio 1 completado!")
        
        return all_passed
    
    return verificar


def test_ejercicio_2_gradient_descent():
    """Test para cálculo de gradientes"""
    def verificar(compute_gradients_func):
        if compute_gradients_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Gradientes básicos
        X1 = np.array([[1, 2], [3, 4], [5, 6]])
        y1 = np.array([5, 11, 17])
        w1 = np.array([1.0, 1.0])
        b1 = 1.0
        
        result = compute_gradients_func(X1, y1, w1, b1)
        
        if result is None:
            print_error("La función retorna None")
            return False
        
        if not isinstance(result, tuple) or len(result) != 2:
            print_error("La función debe retornar una tupla (dw, db)")
            return False
        
        dw1, db1 = result
        
        # Valores esperados calculados manualmente
        # y_pred = X @ w + b = [4, 8, 12]
        # error = y_pred - y = [-1, -3, -5]
        # dw = (1/3) * X.T @ error = (1/3) * [[-1-9-25], [-2-12-30]] = (1/3) * [[-35], [-44]] = [-11.67, -14.67]
        # db = (1/3) * sum(error) = (1/3) * (-9) = -3.0
        
        expected_dw1 = np.array([-35/3, -44/3])
        expected_db1 = -3.0
        
        if dw1 is None:
            print_error("dw es None")
            all_passed = False
        elif not np.allclose(dw1, expected_dw1, atol=1e-10):
            print_error(f"Gradiente dw incorrecto para test 1.")
            print(f"   Expected: {expected_dw1}")
            print(f"   Got: {dw1}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: gradiente dw correcto")
        
        if db1 is None:
            print_error("db es None")
            all_passed = False
        elif not np.isclose(db1, expected_db1, atol=1e-10):
            print_error(f"Gradiente db incorrecto para test 1.")
            print(f"   Expected: {expected_db1}")
            print(f"   Got: {db1}")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: gradiente db correcto")
        
        # Test 2: Caso donde predicción es perfecta (gradientes = 0)
        X2 = np.array([[1, 2], [3, 4]])
        y2 = np.array([5, 11])  # y = 2*x1 + 1*x2 + 0
        w2 = np.array([2.0, 1.0])
        b2 = 0.0
        
        dw2, db2 = compute_gradients_func(X2, y2, w2, b2)
        expected_dw2 = np.array([0.0, 0.0])
        expected_db2 = 0.0
        
        if not np.allclose(dw2, expected_dw2, atol=1e-10):
            print_error(f"Gradiente dw incorrecto para test 2 (predicción perfecta).")
            print(f"   Expected: {expected_dw2}")
            print(f"   Got: {dw2}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: gradientes cero cuando predicción perfecta")
        
        if not np.isclose(db2, expected_db2, atol=1e-10):
            print_error(f"Gradiente db incorrecto para test 2 (predicción perfecta).")
            print(f"   Expected: {expected_db2}")
            print(f"   Got: {db2}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: db cero cuando predicción perfecta")
        
        # Test 3: Verificar shapes
        if dw1.shape != (2,):
            print_error(f"Shape de dw incorrecta. Expected: (2,), Got: {dw1.shape}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: shape de dw correcta")
        
        if not np.isscalar(db1):
            print_error(f"db debe ser escalar, Got: {type(db1)}")
            all_passed = False
        else:
            print("   ✓ Test 6 pasado: db es escalar")
        
        if all_passed:
            print_success("Ejercicio 2 completado!")
        
        return all_passed
    
    return verificar
