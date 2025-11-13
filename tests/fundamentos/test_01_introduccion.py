"""
Tests para Notebook 01: Introducción a Python para Machine Learning
====================================================================

Tests automáticos para verificar las soluciones de los ejercicios.
"""

import numpy as np
import sys
from pathlib import Path

# Agregar utils al path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import test_function, assert_equals, print_success, print_error


def test_ejercicio_1_lista():
    """
    Test para Ejercicio 1: Crear lista y calcular promedio
    """
    def verificar_lista(mi_lista, promedio):
        expected_lista = [10, 11, 12, 13, 14, 15]
        expected_promedio = 12.5

        if mi_lista != expected_lista:
            print_error(f"Lista incorrecta. Expected: {expected_lista}, Got: {mi_lista}")
            return False

        if abs(promedio - expected_promedio) > 1e-7:
            print_error(f"Promedio incorrecto. Expected: {expected_promedio}, Got: {promedio}")
            return False

        print_success("Ejercicio 1 completado correctamente!")
        return True

    return verificar_lista


def test_ejercicio_2_cubos_impares():
    """
    Test para Ejercicio 2: List comprehension con cubos de impares
    """
    expected = [1, 27, 125, 343, 729]

    def verificar(cubos_impares):
        if cubos_impares != expected:
            print_error(f"Lista incorrecta.")
            print(f"   Expected: {expected}")
            print(f"   Got: {cubos_impares}")
            return False

        print_success("Ejercicio 2 completado correctamente!")
        print("   ✓ List comprehension implementada correctamente")
        print("   ✓ Cubos de números impares calculados correctamente")
        return True

    return verificar


def test_ejercicio_3_diccionario():
    """
    Test para Ejercicio 3: Crear diccionario de dataset
    """
    expected = {
        'nombre': 'Iris',
        'n_muestras': 150,
        'n_características': 4,
        'clases': ['setosa', 'versicolor', 'virginica']
    }

    def verificar(dataset_info):
        if not isinstance(dataset_info, dict):
            print_error(f"El tipo debe ser dict, no {type(dataset_info)}")
            return False

        if dataset_info != expected:
            print_error("Diccionario incorrecto.")
            for key in expected:
                if key not in dataset_info:
                    print(f"   ✗ Falta la clave '{key}'")
                elif dataset_info[key] != expected[key]:
                    print(f"   ✗ Valor incorrecto para '{key}'")
                    print(f"      Expected: {expected[key]}")
                    print(f"      Got: {dataset_info[key]}")
            return False

        print_success("Ejercicio 3 completado correctamente!")
        return True

    return verificar


def test_ejercicio_4_varianza():
    """
    Test para Ejercicio 4: Función calcular_varianza
    """
    test_cases = [
        ([1, 2, 3, 4, 5], 2.0),
        ([10, 10, 10, 10], 0.0),
        ([1, 5, 9], 10.666666666666666),
        ([2, 4, 6, 8, 10], 8.0)
    ]

    def run_tests(calcular_varianza):
        all_passed = True

        for i, (input_data, expected) in enumerate(test_cases, 1):
            try:
                result = calcular_varianza(input_data)

                if abs(result - expected) < 1e-7:
                    print(f"   ✓ Test {i} passed")
                else:
                    print_error(f"Test {i} failed")
                    print(f"      Input: {input_data}")
                    print(f"      Expected: {expected}")
                    print(f"      Got: {result}")
                    all_passed = False
            except Exception as e:
                print_error(f"Test {i} raised an exception: {str(e)}")
                all_passed = False

        if all_passed:
            print_success("Ejercicio 4 completado correctamente!")

        return all_passed

    return run_tests


def test_ejercicio_5_clase_calculadora():
    """
    Test para Ejercicio 5: Clase CalculadoraEstadistica
    """
    def run_tests(CalculadoraEstadistica):
        try:
            calc = CalculadoraEstadistica([10, 20, 30, 40, 50])

            tests = [
                ('media', 30.0),
                ('maximo', 50),
                ('minimo', 10)
            ]

            all_passed = True

            for metodo, expected in tests:
                try:
                    result = getattr(calc, metodo)()

                    if abs(result - expected) < 1e-7:
                        print(f"   ✓ Método {metodo}() funciona correctamente")
                    else:
                        print_error(f"Método {metodo}() retorna valor incorrecto")
                        print(f"      Expected: {expected}")
                        print(f"      Got: {result}")
                        all_passed = False
                except AttributeError:
                    print_error(f"Método {metodo}() no existe")
                    all_passed = False
                except Exception as e:
                    print_error(f"Método {metodo}() lanzó una excepción: {str(e)}")
                    all_passed = False

            if all_passed:
                print_success("Ejercicio 5 completado correctamente!")

            return all_passed

        except Exception as e:
            print_error(f"Error al instanciar la clase: {str(e)}")
            return False

    return run_tests


def test_ejercicio_6_celsius_fahrenheit():
    """
    Test para Ejercicio 6: Conversión Celsius a Fahrenheit
    """
    expected = [32.0, 50.0, 68.0, 86.0, 104.0]

    def verificar(fahrenheit):
        # Convertir a lista si es necesario
        if not isinstance(fahrenheit, list):
            fahrenheit = list(fahrenheit)

        if len(fahrenheit) != len(expected):
            print_error(f"Longitud incorrecta. Expected: {len(expected)}, Got: {len(fahrenheit)}")
            return False

        for i, (f, e) in enumerate(zip(fahrenheit, expected)):
            if abs(f - e) > 1e-7:
                print_error(f"Conversión incorrecta en posición {i}")
                print(f"   Expected: {e}°F")
                print(f"   Got: {f}°F")
                return False

        print_success("Ejercicio 6 completado correctamente!")
        print("   ✓ Conversión Celsius → Fahrenheit implementada correctamente")
        return True

    return verificar


# Función de ayuda para casos de prueba
def get_test_cases_varianza():
    """
    Retorna casos de prueba para la función calcular_varianza
    """
    return [
        ([1, 2, 3, 4, 5], 2.0),
        ([10, 10, 10, 10], 0.0),
        ([1, 5, 9], 10.666666666666666)
    ]
