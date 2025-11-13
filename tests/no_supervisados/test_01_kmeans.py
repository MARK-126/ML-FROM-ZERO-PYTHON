"""
Tests para Notebook 01: K-Means Clustering
===========================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_assign_clusters():
    """Test para asignación de puntos a clusters"""
    def verificar(assign_func):
        if assign_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Caso básico
        X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [9, 10]])
        centroids = np.array([[1, 2], [8, 8]])
        
        result = assign_func(X, centroids)
        
        if result is None:
            print_error("La función retorna None")
            return False
        
        # Puntos [1, 2] y [1.5, 1.8] deben ir a centroid 0
        # Puntos [5, 8], [8, 8], [9, 10] deben ir a centroid 1
        expected = np.array([0, 0, 1, 1, 1])
        
        if not np.array_equal(result, expected):
            print_error(f"Asignaciones incorrectas para test 1.")
            print(f"   Expected: {expected}")
            print(f"   Got: {result}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: asignaciones básicas")
        
        # Test 2: Puntos equidistantes
        X2 = np.array([[2.5, 5]])
        centroids2 = np.array([[0, 5], [5, 5]])
        
        result2 = assign_func(X2, centroids2)
        # Punto [2.5, 5] está equidistante, debería ir a 0 (el primero)
        # pero ambos son válidos
        
        if result2[0] not in [0, 1]:
            print_error(f"Asignación debe ser 0 o 1, got: {result2[0]}")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: maneja puntos equidistantes")
        
        # Test 3: Múltiples clusters
        X3 = np.array([[1, 1], [2, 2], [10, 10], [11, 11], [20, 20], [21, 21]])
        centroids3 = np.array([[1.5, 1.5], [10.5, 10.5], [20.5, 20.5]])
        
        result3 = assign_func(X3, centroids3)
        expected3 = np.array([0, 0, 1, 1, 2, 2])
        
        if not np.array_equal(result3, expected3):
            print_error(f"Asignaciones incorrectas para test 3 (múltiples clusters).")
            print(f"   Expected: {expected3}")
            print(f"   Got: {result3}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: múltiples clusters")
        
        # Test 4: Verificar shape
        if result.shape != (5,):
            print_error(f"Shape incorrecta. Expected: (5,), Got: {result.shape}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: shape correcta")
        
        # Test 5: Un solo punto
        X5 = np.array([[3, 4]])
        centroids5 = np.array([[0, 0], [10, 10]])
        
        result5 = assign_func(X5, centroids5)
        expected5 = np.array([0])  # más cercano a [0, 0]
        
        if not np.array_equal(result5, expected5):
            print_error(f"Asignación incorrecta para un solo punto.")
            print(f"   Expected: {expected5}")
            print(f"   Got: {result5}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: un solo punto")
        
        if all_passed:
            print_success("Ejercicio 1 completado!")
        
        return all_passed
    
    return verificar


def test_ejercicio_2_update_centroids():
    """Test para actualización de centroides"""
    def verificar(update_func):
        if update_func is None:
            print_error("La función es None")
            return False
        
        all_passed = True
        
        # Test 1: Caso básico
        X = np.array([[1, 2], [1, 4], [8, 8], [9, 10]])
        labels = np.array([0, 0, 1, 1])
        n_clusters = 2
        
        result = update_func(X, labels, n_clusters)
        
        if result is None:
            print_error("La función retorna None")
            return False
        
        # Cluster 0: media de [[1, 2], [1, 4]] = [1, 3]
        # Cluster 1: media de [[8, 8], [9, 10]] = [8.5, 9]
        expected = np.array([[1., 3.], [8.5, 9.]])
        
        if not np.allclose(result, expected, atol=1e-10):
            print_error(f"Centroides incorrectos para test 1.")
            print(f"   Expected:\n{expected}")
            print(f"   Got:\n{result}")
            all_passed = False
        else:
            print("   ✓ Test 1 pasado: centroides básicos")
        
        # Test 2: Tres clusters
        X2 = np.array([[1, 2], [1, 4], [1, 0], [8, 8], [9, 10], [10, 9], [20, 20], [21, 21]])
        labels2 = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        n_clusters2 = 3
        
        result2 = update_func(X2, labels2, n_clusters2)
        
        # Cluster 0: media de [[1,2], [1,4], [1,0]] = [1, 2]
        # Cluster 1: media de [[8,8], [9,10], [10,9]] = [9, 9]
        # Cluster 2: media de [[20,20], [21,21]] = [20.5, 20.5]
        expected2 = np.array([[1., 2.], [9., 9.], [20.5, 20.5]])
        
        if not np.allclose(result2, expected2, atol=1e-10):
            print_error(f"Centroides incorrectos para test 2 (tres clusters).")
            print(f"   Expected:\n{expected2}")
            print(f"   Got:\n{result2}")
            all_passed = False
        else:
            print("   ✓ Test 2 pasado: tres clusters")
        
        # Test 3: Verificar shape
        if result.shape != (2, 2):
            print_error(f"Shape incorrecta. Expected: (2, 2), Got: {result.shape}")
            all_passed = False
        else:
            print("   ✓ Test 3 pasado: shape correcta")
        
        # Test 4: Un punto por cluster
        X4 = np.array([[1, 1], [5, 5], [10, 10]])
        labels4 = np.array([0, 1, 2])
        n_clusters4 = 3
        
        result4 = update_func(X4, labels4, n_clusters4)
        expected4 = np.array([[1., 1.], [5., 5.], [10., 10.]])
        
        if not np.allclose(result4, expected4, atol=1e-10):
            print_error(f"Centroides incorrectos para test 4 (un punto por cluster).")
            print(f"   Expected:\n{expected4}")
            print(f"   Got:\n{result4}")
            all_passed = False
        else:
            print("   ✓ Test 4 pasado: un punto por cluster")
        
        # Test 5: Cluster con múltiples puntos
        X5 = np.array([[0, 0], [2, 0], [0, 2], [2, 2]])
        labels5 = np.array([0, 0, 0, 0])
        n_clusters5 = 1
        
        result5 = update_func(X5, labels5, n_clusters5)
        expected5 = np.array([[1., 1.]])  # media de todos los puntos
        
        if not np.allclose(result5, expected5, atol=1e-10):
            print_error(f"Centroides incorrectos para test 5 (todos mismo cluster).")
            print(f"   Expected:\n{expected5}")
            print(f"   Got:\n{result5}")
            all_passed = False
        else:
            print("   ✓ Test 5 pasado: todos en mismo cluster")
        
        if all_passed:
            print_success("Ejercicio 2 completado!")
        
        return all_passed
    
    return verificar
