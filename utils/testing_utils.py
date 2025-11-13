"""
Utilidades de Testing para Notebooks de ML desde Cero
======================================================

Este módulo contiene funciones para verificar respuestas de ejercicios
y proporcionar feedback automático a los estudiantes.

Autor: ML desde Cero
Versión: 1.0
"""

import numpy as np
from typing import Any, List, Tuple, Dict, Callable


def check_value(student_answer: Any, expected_answer: Any, tolerance: float = 1e-7) -> bool:
    """
    Verifica si dos valores son iguales (con tolerancia para floats).

    Arguments:
    ----------
    student_answer : Any
        Respuesta del estudiante
    expected_answer : Any
        Respuesta esperada
    tolerance : float
        Tolerancia para comparación de números flotantes

    Returns:
    --------
    bool
        True si las respuestas son iguales, False en caso contrario
    """
    if isinstance(expected_answer, (int, float)) and isinstance(student_answer, (int, float)):
        return abs(student_answer - expected_answer) < tolerance
    elif isinstance(expected_answer, np.ndarray) and isinstance(student_answer, np.ndarray):
        return np.allclose(student_answer, expected_answer, atol=tolerance)
    elif isinstance(expected_answer, list) and isinstance(student_answer, list):
        if len(student_answer) != len(expected_answer):
            return False
        return all(check_value(s, e, tolerance) for s, e in zip(student_answer, expected_answer))
    elif isinstance(expected_answer, dict) and isinstance(student_answer, dict):
        if set(student_answer.keys()) != set(expected_answer.keys()):
            return False
        return all(check_value(student_answer[k], expected_answer[k], tolerance)
                   for k in expected_answer.keys())
    else:
        return student_answer == expected_answer


def check_type(value: Any, expected_type: type) -> bool:
    """
    Verifica si un valor es del tipo esperado.

    Arguments:
    ----------
    value : Any
        Valor a verificar
    expected_type : type
        Tipo esperado

    Returns:
    --------
    bool
        True si el tipo es correcto, False en caso contrario
    """
    return isinstance(value, expected_type)


def check_shape(array: np.ndarray, expected_shape: tuple) -> bool:
    """
    Verifica si un array tiene la forma esperada.

    Arguments:
    ----------
    array : np.ndarray
        Array a verificar
    expected_shape : tuple
        Forma esperada

    Returns:
    --------
    bool
        True si la forma es correcta, False en caso contrario
    """
    return array.shape == expected_shape


def single_test(test_case: Dict, target_function: Callable) -> bool:
    """
    Ejecuta un caso de prueba individual.

    Arguments:
    ----------
    test_case : Dict
        Diccionario con 'name', 'input', 'expected' y 'error'
    target_function : Callable
        Función a probar

    Returns:
    --------
    bool
        True si la prueba pasa, False en caso contrario
    """
    try:
        if isinstance(test_case['input'], list):
            result = target_function(*test_case['input'])
        else:
            result = target_function(test_case['input'])

        test_name = test_case['name']

        if test_name == 'datatype_check':
            if isinstance(test_case['expected'], list):
                for i, (res, exp) in enumerate(zip(result, test_case['expected'])):
                    if not check_type(res, type(exp)):
                        print(f"\033[91m❌ {test_case['error']}: {type(res)} != {type(exp)}")
                        return False
            else:
                if not check_type(result, type(test_case['expected'])):
                    print(f"\033[91m❌ {test_case['error']}: {type(result)} != {type(test_case['expected'])}")
                    return False

        elif test_name == 'shape_check':
            if isinstance(result, np.ndarray):
                expected_shape = test_case['expected'].shape if isinstance(test_case['expected'], np.ndarray) else test_case['expected']
                if not check_shape(result, expected_shape):
                    print(f"\033[91m❌ {test_case['error']}: {result.shape} != {expected_shape}")
                    return False

        elif test_name == 'equation_output_check':
            if not check_value(result, test_case['expected']):
                print(f"\033[91m❌ {test_case['error']}")
                print(f"   Expected: {test_case['expected']}")
                print(f"   Got: {result}")
                return False

        return True

    except Exception as e:
        print(f"\033[91m❌ Error durante la prueba: {str(e)}")
        return False


def multiple_test(test_cases: List[Dict], target_function: Callable) -> None:
    """
    Ejecuta múltiples casos de prueba.

    Arguments:
    ----------
    test_cases : List[Dict]
        Lista de casos de prueba
    target_function : Callable
        Función a probar
    """
    all_passed = True
    for test_case in test_cases:
        if not single_test(test_case, target_function):
            all_passed = False

    if all_passed:
        print("\033[92m✅ All tests passed!")


def test_function(func: Callable, test_cases: List[Tuple], func_name: str = None) -> None:
    """
    Prueba una función con múltiples casos de prueba.

    Arguments:
    ----------
    func : Callable
        Función a probar
    test_cases : List[Tuple]
        Lista de tuplas (inputs, expected_output)
    func_name : str
        Nombre de la función (opcional)
    """
    name = func_name or func.__name__
    all_passed = True

    for i, (inputs, expected) in enumerate(test_cases, 1):
        try:
            if isinstance(inputs, tuple):
                result = func(*inputs)
            else:
                result = func(inputs)

            if not check_value(result, expected):
                all_passed = False
                print(f"\033[91m❌ Test {i} failed for {name}")
                print(f"   Input: {inputs}")
                print(f"   Expected: {expected}")
                print(f"   Got: {result}")
        except Exception as e:
            all_passed = False
            print(f"\033[91m❌ Test {i} raised an exception for {name}")
            print(f"   Input: {inputs}")
            print(f"   Error: {str(e)}")

    if all_passed:
        print(f"\033[92m✅ All tests passed for {name}!")


def assert_equals(actual: Any, expected: Any, message: str = "") -> None:
    """
    Verifica que dos valores sean iguales y muestra un mensaje.

    Arguments:
    ----------
    actual : Any
        Valor actual
    expected : Any
        Valor esperado
    message : str
        Mensaje adicional
    """
    if check_value(actual, expected):
        print(f"\033[92m✅ Test passed! {message}")
    else:
        print(f"\033[91m❌ Test failed! {message}")
        print(f"   Expected: {expected}")
        print(f"   Got: {actual}")
        raise AssertionError(f"Expected {expected}, but got {actual}")


def assert_array_equals(actual: np.ndarray, expected: np.ndarray, message: str = "") -> None:
    """
    Verifica que dos arrays sean iguales.

    Arguments:
    ----------
    actual : np.ndarray
        Array actual
    expected : np.ndarray
        Array esperado
    message : str
        Mensaje adicional
    """
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        if np.allclose(actual, expected):
            print(f"\033[92m✅ Test passed! {message}")
        else:
            print(f"\033[91m❌ Test failed! {message}")
            print(f"   Expected:\n{expected}")
            print(f"   Got:\n{actual}")
            raise AssertionError("Arrays are not equal")
    else:
        print(f"\033[91m❌ Type error: Expected numpy arrays")
        raise TypeError("Both inputs must be numpy arrays")


def assert_shape_equals(array: np.ndarray, expected_shape: tuple, message: str = "") -> None:
    """
    Verifica que un array tenga la forma esperada.

    Arguments:
    ----------
    array : np.ndarray
        Array a verificar
    expected_shape : tuple
        Forma esperada
    message : str
        Mensaje adicional
    """
    if check_shape(array, expected_shape):
        print(f"\033[92m✅ Shape test passed! {message}")
    else:
        print(f"\033[91m❌ Shape test failed! {message}")
        print(f"   Expected shape: {expected_shape}")
        print(f"   Got shape: {array.shape}")
        raise AssertionError(f"Expected shape {expected_shape}, but got {array.shape}")


def print_success(message: str = "All tests passed!") -> None:
    """
    Imprime un mensaje de éxito en verde.

    Arguments:
    ----------
    message : str
        Mensaje a imprimir
    """
    print(f"\033[92m✅ {message}")


def print_error(message: str) -> None:
    """
    Imprime un mensaje de error en rojo.

    Arguments:
    ----------
    message : str
        Mensaje a imprimir
    """
    print(f"\033[91m❌ {message}")


def print_info(message: str) -> None:
    """
    Imprime un mensaje informativo en azul.

    Arguments:
    ----------
    message : str
        Mensaje a imprimir
    """
    print(f"\033[94mℹ️  {message}")


def print_warning(message: str) -> None:
    """
    Imprime un mensaje de advertencia en amarillo.

    Arguments:
    ----------
    message : str
        Mensaje a imprimir
    """
    print(f"\033[93m⚠️  {message}")
