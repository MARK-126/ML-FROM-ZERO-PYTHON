"""
Tests para Notebook 05: Fundamentos de Machine Learning
=========================================================
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_train_test_split():
    """Test para función train_test_split"""
    def verificar(split_func):
        # Crear datos de prueba
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)
        
        # Test 1: Verificar que retorna 4 valores
        result = split_func(X, y, test_size=0.2)
        if result is None or len(result) != 4:
            print_error("La función debe retornar 4 valores: (X_train, X_test, y_train, y_test)")
            return False
        
        X_train, X_test, y_train, y_test = result
        
        if X_train is None or X_test is None or y_train is None or y_test is None:
            print_error("Uno de los valores retornados es None")
            return False
        
        all_passed = True
        
        # Test 2: Verificar tamaños
        n_total = len(X)
        expected_test_size = int(n_total * 0.2)
        expected_train_size = n_total - expected_test_size
        
        if len(X_train) != expected_train_size:
            print_error(f"Tamaño de X_train incorrecto. Expected: {expected_train_size}, Got: {len(X_train)}")
            all_passed = False
        else:
            print(f"   ✓ Tamaño de train set correcto: {len(X_train)}")
        
        if len(X_test) != expected_test_size:
            print_error(f"Tamaño de X_test incorrecto. Expected: {expected_test_size}, Got: {len(X_test)}")
            all_passed = False
        else:
            print(f"   ✓ Tamaño de test set correcto: {len(X_test)}")
        
        # Test 3: Verificar que y tiene los mismos tamaños
        if len(y_train) != len(X_train):
            print_error(f"Tamaño de y_train no coincide con X_train")
            all_passed = False
        else:
            print("   ✓ Tamaños de y coinciden con X")
        
        # Test 4: Verificar que no hay duplicados entre train y test
        X_train_flat = X_train.flatten()
        X_test_flat = X_test.flatten()
        
        # Verificar que no hay overlap
        overlap = np.intersect1d(X_train_flat, X_test_flat)
        if len(overlap) > 0:
            print_error(f"Hay {len(overlap)} elementos duplicados entre train y test")
            all_passed = False
        else:
            print("   ✓ No hay overlap entre train y test")
        
        # Test 5: Verificar que todos los datos están presentes
        all_data = np.concatenate([X_train_flat, X_test_flat])
        all_data_sorted = np.sort(all_data)
        original_sorted = np.sort(X.flatten())
        
        if not np.array_equal(all_data_sorted, original_sorted):
            print_error("No se preservaron todos los datos")
            all_passed = False
        else:
            print("   ✓ Todos los datos se preservaron")
        
        if all_passed:
            print_success("Ejercicio 1 completado!")
        
        return all_passed
    
    return verificar


def test_ejercicio_2_mse():
    """Test para función calcular_mse"""
    def verificar(mse_func):
        all_passed = True
        
        # Test 1: MSE con predicciones perfectas
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        mse = mse_func(y_true, y_pred)
        
        if mse is None:
            print_error("La función retorna None")
            return False
        
        if not np.isclose(mse, 0.0, atol=1e-10):
            print_error(f"MSE con predicciones perfectas debería ser 0. Got: {mse}")
            all_passed = False
        else:
            print("   ✓ MSE con predicciones perfectas = 0")
        
        # Test 2: MSE con valores conocidos
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        # MSE esperado: mean([0.01, 0.01, 0.01, 0.04, 0.04]) = 0.022
        expected_mse = 0.022
        mse = mse_func(y_true, y_pred)
        
        if not np.isclose(mse, expected_mse, atol=1e-6):
            print_error(f"MSE incorrecto. Expected: {expected_mse:.6f}, Got: {mse:.6f}")
            all_passed = False
        else:
            print(f"   ✓ MSE calculado correctamente: {mse:.6f}")
        
        # Test 3: MSE con errores más grandes
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 32.0])
        
        # MSE = mean([1, 1, 4]) = 2.0
        expected_mse = 2.0
        mse = mse_func(y_true, y_pred)
        
        if not np.isclose(mse, expected_mse, atol=1e-10):
            print_error(f"MSE incorrecto para errores grandes. Expected: {expected_mse}, Got: {mse}")
            all_passed = False
        else:
            print(f"   ✓ MSE con errores grandes correcto: {mse}")
        
        # Test 4: Verificar que usa operaciones vectorizadas (no bucles)
        # Este test verifica que la función funciona con arrays grandes eficientemente
        y_true_large = np.random.randn(10000)
        y_pred_large = y_true_large + np.random.randn(10000) * 0.1
        
        try:
            mse_large = mse_func(y_true_large, y_pred_large)
            if mse_large is not None and mse_large >= 0:
                print("   ✓ Función funciona con arrays grandes")
            else:
                print_error("MSE con arrays grandes retorna valor inválido")
                all_passed = False
        except Exception as e:
            print_error(f"Error con arrays grandes: {str(e)}")
            all_passed = False
        
        if all_passed:
            print_success("Ejercicio 2 completado!")
        
        return all_passed
    
    return verificar
