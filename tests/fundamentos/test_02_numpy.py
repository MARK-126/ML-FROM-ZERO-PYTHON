"""
Tests para Notebook 02: NumPy - Operaciones con Arrays
=======================================================

Tests automáticos para verificar las soluciones de los ejercicios.
"""

import numpy as np
import sys
from pathlib import Path

# Agregar utils al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import assert_array_equals, assert_shape_equals, print_success, print_error


def test_ejercicio_1_crear_arrays():
    """
    Test para Ejercicio 1: Crear arrays básicos
    """
    def verificar(arr_1, arr_2, arr_3):
        all_passed = True

        # Test array 1
        expected_1 = np.arange(10, 21)
        if not np.array_equal(arr_1, expected_1):
            print_error("Array 1 incorrecto")
            print(f"   Expected: {expected_1}")
            print(f"   Got: {arr_1}")
            all_passed = False
        else:
            print("   ✓ Array del 10 al 20 correcto")

        # Test array 2
        expected_2 = np.eye(4)
        if not np.allclose(arr_2, expected_2):
            print_error("Array 2 (matriz identidad) incorrecto")
            all_passed = False
        else:
            print("   ✓ Matriz identidad 4x4 correcta")

        # Test array 3
        expected_3 = np.ones((3, 5))
        if not np.array_equal(arr_3, expected_3):
            print_error("Array 3 (matriz de unos) incorrecto")
            all_passed = False
        else:
            print("   ✓ Matriz 3x5 de unos correcta")

        if all_passed:
            print_success("Ejercicio 1 completado correctamente!")

        return all_passed

    return verificar


def test_ejercicio_2_indexacion():
    """
    Test para Ejercicio 2: Indexación y slicing
    """
    def verificar(tercera_columna, submatriz, mayores_60):
        all_passed = True

        expected_col = np.array([30, 70, 110])
        expected_sub = np.array([[70, 80], [110, 120]])
        expected_mayores = np.array([70, 80, 90, 100, 110, 120])

        if not np.array_equal(tercera_columna, expected_col):
            print_error("Tercera columna incorrecta")
            print(f"   Expected: {expected_col}")
            print(f"   Got: {tercera_columna}")
            all_passed = False
        else:
            print("   ✓ Tercera columna extraída correctamente")

        if not np.array_equal(submatriz, expected_sub):
            print_error("Submatriz incorrecta")
            all_passed = False
        else:
            print("   ✓ Submatriz extraída correctamente")

        if not np.array_equal(mayores_60, expected_mayores):
            print_error("Elementos mayores a 60 incorrectos")
            print(f"   Expected: {expected_mayores}")
            print(f"   Got: {mayores_60}")
            all_passed = False
        else:
            print("   ✓ Filtrado booleano correcto")

        if all_passed:
            print_success("Ejercicio 2 completado correctamente!")

        return all_passed

    return verificar


def test_ejercicio_3_broadcasting():
    """
    Test para Ejercicio 3: Broadcasting y normalización
    """
    def verificar(datos_normalizados):
        if datos_normalizados is None:
            print_error("datos_normalizados es None")
            return False

        all_passed = True

        # Verificar que la media sea ~0
        media = np.mean(datos_normalizados, axis=0)
        if not np.allclose(media, np.zeros(3), atol=1e-10):
            print_error(f"Media incorrecta. Expected: [0, 0, 0], Got: {media}")
            all_passed = False
        else:
            print("   ✓ Media de columnas = 0")

        # Verificar que la std sea ~1
        std = np.std(datos_normalizados, axis=0)
        if not np.allclose(std, np.ones(3), atol=1e-10):
            print_error(f"Desviación estándar incorrecta. Expected: [1, 1, 1], Got: {std}")
            all_passed = False
        else:
            print("   ✓ Desviación estándar de columnas = 1")

        if all_passed:
            print_success("Ejercicio 3 completado correctamente!")
            print("   Broadcasting aplicado correctamente")

        return all_passed

    return verificar


def test_ejercicio_4_reshape():
    """
    Test para Ejercicio 4: Reshape y transpuesta
    """
    def verificar(matriz, transpuesta, suma_filas):
        all_passed = True

        if matriz is None:
            print_error("matriz es None")
            return False

        if matriz.shape != (4, 5):
            print_error(f"Shape de matriz incorrecto. Expected: (4, 5), Got: {matriz.shape}")
            all_passed = False
        else:
            print("   ✓ Reshape a 4x5 correcto")

        if transpuesta is None:
            print_error("transpuesta es None")
            return False

        if transpuesta.shape != (5, 4):
            print_error(f"Shape de transpuesta incorrecto. Expected: (5, 4), Got: {transpuesta.shape}")
            all_passed = False
        else:
            print("   ✓ Transpuesta calculada correctamente")

        if suma_filas is None:
            print_error("suma_filas es None")
            return False

        expected_suma = np.array([50, 54, 58, 62, 66])
        if not np.array_equal(suma_filas, expected_suma):
            print_error("Suma de filas incorrecta")
            print(f"   Expected: {expected_suma}")
            print(f"   Got: {suma_filas}")
            all_passed = False
        else:
            print("   ✓ Suma de filas correcta")

        if all_passed:
            print_success("Ejercicio 4 completado correctamente!")

        return all_passed

    return verificar


def test_ejercicio_5_algebra():
    """
    Test para Ejercicio 5: Álgebra lineal
    """
    def verificar(producto, determinante):
        if producto is None:
            print_error("producto es None")
            return False

        if determinante is None:
            print_error("determinante es None")
            return False

        all_passed = True

        expected_producto = np.array([[2, 2], [1, 6]])
        if not np.array_equal(producto, expected_producto):
            print_error("Producto matricial incorrecto")
            print(f"   Expected:\n{expected_producto}")
            print(f"   Got:\n{producto}")
            all_passed = False
        else:
            print("   ✓ Producto matricial A @ B correcto")

        if not np.isclose(determinante, 10.0, atol=1e-10):
            print_error(f"Determinante incorrecto. Expected: 10.0, Got: {determinante}")
            all_passed = False
        else:
            print("   ✓ Determinante calculado correctamente")

        if all_passed:
            print_success("Ejercicio 5 completado correctamente!")

        return all_passed

    return verificar


def test_ejercicio_6_minmax():
    """
    Test para Ejercicio 6: Normalización Min-Max
    """
    def verificar(resultado):
        if resultado is None:
            print_error("resultado es None")
            return False

        all_passed = True

        # Verificar que los valores estén en [0, 1]
        min_vals = np.min(resultado, axis=0)
        max_vals = np.max(resultado, axis=0)

        if not np.allclose(min_vals, np.zeros(2), atol=1e-10):
            print_error(f"Mínimos incorrectos. Expected: [0, 0], Got: {min_vals}")
            all_passed = False
        else:
            print("   ✓ Valores mínimos = 0")

        if not np.allclose(max_vals, np.ones(2), atol=1e-10):
            print_error(f"Máximos incorrectos. Expected: [1, 1], Got: {max_vals}")
            all_passed = False
        else:
            print("   ✓ Valores máximos = 1")

        # Verificar normalización correcta
        expected = np.array([[0.0, 0.0],
                             [1/3, 1/3],
                             [2/3, 2/3],
                             [1.0, 1.0]])

        if not np.allclose(resultado, expected, atol=1e-10):
            print_error("Normalización incorrecta")
            print(f"   Expected:\n{expected}")
            print(f"   Got:\n{resultado}")
            all_passed = False
        else:
            print("   ✓ Normalización Min-Max correcta")

        if all_passed:
            print_success("Ejercicio 6 completado correctamente!")

        return all_passed

    return verificar
