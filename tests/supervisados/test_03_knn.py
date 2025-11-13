"""
Tests para Notebook 03: K-Nearest Neighbors
============================================
"""

import numpy as np
import sys
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_distancia():
    """Test para función de cálculo de distancias"""
    def verificar(distances_func):
        if distances_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Distancias básicas
        x = np.array([0, 0])
        X_train = np.array([[1, 0], [0, 1], [3, 4]])
        
        result = distances_func(x, X_train)
        expected = np.array([1.0, 1.0, 5.0])
        
        if result is None:
            print_error("La función retorna None")
            return False
        
        if not np.allclose(result, expected, atol=1e-10):
            print_error(f"Distancias incorrectas para test 1.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: distancias básicas")
        
        # Test 2: Punto idéntico (distancia 0)
        x2 = np.array([2, 3])
        X_train2 = np.array([[2, 3], [5, 7], [1, 1]])
        
        result2 = distances_func(x2, X_train2)
        expected2 = np.array([0.0, 5.0, np.sqrt(5)])
        
        if not np.allclose(result2, expected2, atol=1e-10):
            print_error(f"Distancias incorrectas para test 2 (punto idéntico).")
            print(f"   Expected: {expected2}")
            print(f"   Got: {result2}")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: punto idéntico (distancia 0)")
        
        # Test 3: Verificar shape
        if result.shape != (3,):
            print_error(f"Shape incorrecta. Expected: (3,), Got: {result.shape}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: shape correcta")
        
        # Test 4: Múltiples dimensiones
        x3 = np.array([1, 2, 3])
        X_train3 = np.array([[1, 2, 3], [4, 5, 6], [0, 0, 0]])
        
        result3 = distances_func(x3, X_train3)
        expected3 = np.array([0.0, np.sqrt(27), np.sqrt(14)])
        
        if not np.allclose(result3, expected3, atol=1e-10):
            print_error(f"Distancias incorrectas para test 3 (3D).")
            print(f"   Expected: {expected3}")
            print(f"   Got: {result3}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: múltiples dimensiones")
        
        # Test 5: Verificar que usa operaciones vectorizadas (no bucles)
        # Si es vectorizado, debería funcionar con arrays grandes eficientemente
        x_large = np.random.rand(10)
        X_large = np.random.rand(100, 10)
        try:
            result_large = distances_func(x_large, X_large)
            if result_large.shape == (100,):
                print("   ✓ Test 5 pasado: funciona con arrays grandes")
            else:
                print_error(f"Shape incorrecta para arrays grandes: {result_large.shape}")
                all_passed = False
        except Exception as e:
            print_error(f"Error con arrays grandes: {e}")
            all_passed = False
        
        if all_passed:
            print_success("Ejercicio 1 completado!")
        
        return all_passed
    
    return verificar


def test_ejercicio_2_knn_predict():
    """Test para función de predicción KNN"""
    def verificar(predict_func):
        if predict_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Caso simple K=1
        X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        x_test = np.array([3, 3])
        
        result = predict_func(x_test, X_train, y_train, k=1)
        
        if result is None:
            print_error("La función retorna None")
            return False
        
        # El punto más cercano a [3, 3] es [2, 3] con clase 0
        expected = 0
        
        if result != expected:
            print_error(f"Predicción incorrecta para test 1 (K=1).")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: K=1 correcto")
        
        # Test 2: Caso K=3
        result2 = predict_func(x_test, X_train, y_train, k=3)
        # Los 3 vecinos más cercanos a [3, 3] son [2, 3], [3, 1], [1, 2] todos clase 0
        expected2 = 0
        
        if result2 != expected2:
            print_error(f"Predicción incorrecta para test 2 (K=3).")
            print(f"   Expected: {expected2}")
            print(f"   Got: {result2}")
            print("   Hint: Verifica que estés ordenando correctamente las distancias")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: K=3 correcto")
        
        # Test 3: Punto cercano a clase 1
        x_test3 = np.array([7, 6])
        result3 = predict_func(x_test3, X_train, y_train, k=3)
        expected3 = 1
        
        if result3 != expected3:
            print_error(f"Predicción incorrecta para test 3.")
            print(f"   Expected: {expected3}")
            print(f"   Got: {result3}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: predice clase 1 correctamente")
        
        # Test 4: Votación con empate (debe resolver)
        X_train4 = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y_train4 = np.array([0, 1, 0, 1])
        x_test4 = np.array([1.5, 1.5])
        
        result4 = predict_func(x_test4, X_train4, y_train4, k=2)
        # Los 2 más cercanos son [1, 1] y [2, 2] con clases 1 y 0
        # Debería retornar alguna clase (0 o 1)
        
        if result4 not in [0, 1]:
            print_error(f"Predicción debe ser 0 o 1, got: {result4}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: maneja empates")
        
        # Test 5: K=5 con mayoría clara
        X_train5 = np.array([[i, i] for i in range(10)])
        y_train5 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        x_test5 = np.array([2, 2])
        
        result5 = predict_func(x_test5, X_train5, y_train5, k=5)
        # Los 5 vecinos más cercanos son [0,0], [1,1], [2,2], [3,3], [4,4] - todos clase 0
        expected5 = 0
        
        if result5 != expected5:
            print_error(f"Predicción incorrecta para test 5 (K=5).")
            print(f"   Expected: {expected5}")
            print(f"   Got: {result5}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: K=5 con mayoría clara")
        
        if all_passed:
            print_success("Ejercicio 2 completado!")
        
        return all_passed
    
    return verificar
