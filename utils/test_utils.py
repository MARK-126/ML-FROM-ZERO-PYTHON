"""
Utilidades para verificar respuestas de ejercicios
"""
import numpy as np
from typing import Any, Callable, Union
import inspect


class TestResult:
    """Clase para representar el resultado de un test"""

    def __init__(self, passed: bool, message: str):
        self.passed = passed
        self.message = message

    def __repr__(self):
        if self.passed:
            return f"✅ TEST PASADO: {self.message}"
        else:
            return f"❌ TEST FALLIDO: {self.message}"


def check_answer(student_answer: Any, expected_answer: Any,
                 tolerance: float = 1e-5, answer_type: str = "value") -> TestResult:
    """
    Verifica si la respuesta del estudiante es correcta.

    Parámetros:
    -----------
    student_answer : Any
        La respuesta del estudiante
    expected_answer : Any
        La respuesta esperada
    tolerance : float
        Tolerancia para comparaciones numéricas
    answer_type : str
        Tipo de respuesta: 'value', 'array', 'string', 'type'

    Returns:
    --------
    TestResult
        Resultado del test
    """
    try:
        if answer_type == "value":
            if isinstance(expected_answer, (int, float, np.number)):
                if abs(student_answer - expected_answer) <= tolerance:
                    return TestResult(True, f"Respuesta correcta: {student_answer}")
                else:
                    return TestResult(False,
                                    f"Respuesta incorrecta. Esperado: {expected_answer}, "
                                    f"Obtenido: {student_answer}")
            else:
                if student_answer == expected_answer:
                    return TestResult(True, f"Respuesta correcta: {student_answer}")
                else:
                    return TestResult(False,
                                    f"Respuesta incorrecta. Esperado: {expected_answer}, "
                                    f"Obtenido: {student_answer}")

        elif answer_type == "array":
            return check_array(student_answer, expected_answer, tolerance)

        elif answer_type == "string":
            if str(student_answer).strip().lower() == str(expected_answer).strip().lower():
                return TestResult(True, "Respuesta correcta")
            else:
                return TestResult(False,
                                f"Respuesta incorrecta. Esperado: {expected_answer}, "
                                f"Obtenido: {student_answer}")

        elif answer_type == "type":
            if type(student_answer) == expected_answer:
                return TestResult(True, f"Tipo correcto: {type(student_answer).__name__}")
            else:
                return TestResult(False,
                                f"Tipo incorrecto. Esperado: {expected_answer.__name__}, "
                                f"Obtenido: {type(student_answer).__name__}")

        else:
            return TestResult(False, f"Tipo de test no reconocido: {answer_type}")

    except Exception as e:
        return TestResult(False, f"Error al verificar la respuesta: {str(e)}")


def check_array(student_array: np.ndarray, expected_array: np.ndarray,
                tolerance: float = 1e-5) -> TestResult:
    """
    Verifica si un array de NumPy es correcto.

    Parámetros:
    -----------
    student_array : np.ndarray
        Array del estudiante
    expected_array : np.ndarray
        Array esperado
    tolerance : float
        Tolerancia para la comparación

    Returns:
    --------
    TestResult
        Resultado del test
    """
    try:
        # Convertir a numpy array si no lo es
        student_array = np.asarray(student_array)
        expected_array = np.asarray(expected_array)

        # Verificar forma
        if student_array.shape != expected_array.shape:
            return TestResult(False,
                            f"Forma incorrecta. Esperado: {expected_array.shape}, "
                            f"Obtenido: {student_array.shape}")

        # Verificar valores
        if np.allclose(student_array, expected_array, atol=tolerance, rtol=tolerance):
            return TestResult(True, "Array correcto")
        else:
            max_diff = np.max(np.abs(student_array - expected_array))
            return TestResult(False,
                            f"Valores incorrectos. Diferencia máxima: {max_diff:.6f}")

    except Exception as e:
        return TestResult(False, f"Error al verificar el array: {str(e)}")


def check_function(student_function: Callable, test_cases: list,
                   tolerance: float = 1e-5) -> TestResult:
    """
    Verifica si una función implementada por el estudiante es correcta.

    Parámetros:
    -----------
    student_function : Callable
        Función del estudiante
    test_cases : list
        Lista de tuplas (input, expected_output)
    tolerance : float
        Tolerancia para comparaciones numéricas

    Returns:
    --------
    TestResult
        Resultado del test
    """
    try:
        if not callable(student_function):
            return TestResult(False, "El objeto proporcionado no es una función")

        for i, (test_input, expected_output) in enumerate(test_cases):
            # Ejecutar la función con el input de prueba
            if isinstance(test_input, tuple):
                result = student_function(*test_input)
            else:
                result = student_function(test_input)

            # Verificar el resultado
            if isinstance(expected_output, np.ndarray):
                if not np.allclose(result, expected_output, atol=tolerance, rtol=tolerance):
                    return TestResult(False,
                                    f"Test case {i+1} fallido. Input: {test_input}, "
                                    f"Esperado: {expected_output}, Obtenido: {result}")
            elif isinstance(expected_output, (int, float)):
                if abs(result - expected_output) > tolerance:
                    return TestResult(False,
                                    f"Test case {i+1} fallido. Input: {test_input}, "
                                    f"Esperado: {expected_output}, Obtenido: {result}")
            else:
                if result != expected_output:
                    return TestResult(False,
                                    f"Test case {i+1} fallido. Input: {test_input}, "
                                    f"Esperado: {expected_output}, Obtenido: {result}")

        return TestResult(True, f"Todos los {len(test_cases)} test cases pasaron correctamente")

    except Exception as e:
        return TestResult(False, f"Error al ejecutar la función: {str(e)}")


def check_model(model, required_methods: list = None,
                required_attributes: list = None) -> TestResult:
    """
    Verifica si un modelo de ML tiene los métodos y atributos requeridos.

    Parámetros:
    -----------
    model : object
        Modelo a verificar
    required_methods : list
        Lista de nombres de métodos que debe tener el modelo
    required_attributes : list
        Lista de nombres de atributos que debe tener el modelo

    Returns:
    --------
    TestResult
        Resultado del test
    """
    try:
        if required_methods is None:
            required_methods = ['fit', 'predict']

        if required_attributes is None:
            required_attributes = []

        # Verificar métodos
        for method in required_methods:
            if not hasattr(model, method):
                return TestResult(False, f"El modelo no tiene el método '{method}'")
            if not callable(getattr(model, method)):
                return TestResult(False, f"'{method}' no es un método callable")

        # Verificar atributos
        for attr in required_attributes:
            if not hasattr(model, attr):
                return TestResult(False, f"El modelo no tiene el atributo '{attr}'")

        return TestResult(True, "El modelo tiene todos los métodos y atributos requeridos")

    except Exception as e:
        return TestResult(False, f"Error al verificar el modelo: {str(e)}")


def check_shape(array: np.ndarray, expected_shape: tuple) -> TestResult:
    """
    Verifica si un array tiene la forma esperada.

    Parámetros:
    -----------
    array : np.ndarray
        Array a verificar
    expected_shape : tuple
        Forma esperada

    Returns:
    --------
    TestResult
        Resultado del test
    """
    try:
        array = np.asarray(array)
        if array.shape == expected_shape:
            return TestResult(True, f"Forma correcta: {array.shape}")
        else:
            return TestResult(False,
                            f"Forma incorrecta. Esperado: {expected_shape}, "
                            f"Obtenido: {array.shape}")
    except Exception as e:
        return TestResult(False, f"Error al verificar la forma: {str(e)}")


def check_range(value: float, min_value: float, max_value: float,
                inclusive: bool = True) -> TestResult:
    """
    Verifica si un valor está dentro de un rango.

    Parámetros:
    -----------
    value : float
        Valor a verificar
    min_value : float
        Valor mínimo del rango
    max_value : float
        Valor máximo del rango
    inclusive : bool
        Si True, incluye los extremos del rango

    Returns:
    --------
    TestResult
        Resultado del test
    """
    try:
        if inclusive:
            if min_value <= value <= max_value:
                return TestResult(True, f"Valor {value} está en el rango [{min_value}, {max_value}]")
            else:
                return TestResult(False,
                                f"Valor {value} está fuera del rango [{min_value}, {max_value}]")
        else:
            if min_value < value < max_value:
                return TestResult(True, f"Valor {value} está en el rango ({min_value}, {max_value})")
            else:
                return TestResult(False,
                                f"Valor {value} está fuera del rango ({min_value}, {max_value})")
    except Exception as e:
        return TestResult(False, f"Error al verificar el rango: {str(e)}")


def print_test_summary(results: list):
    """
    Imprime un resumen de los resultados de múltiples tests.

    Parámetros:
    -----------
    results : list
        Lista de TestResult
    """
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "="*60)
    print(f"RESUMEN DE TESTS: {passed}/{total} pasados")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"\nTest {i}: {result}")

    if passed == total:
        print("\n🎉 ¡EXCELENTE! Todos los tests pasaron.")
    elif passed > 0:
        print(f"\n⚠️  {total - passed} test(s) fallaron. Revisa tu código.")
    else:
        print("\n❌ Todos los tests fallaron. Revisa tu código cuidadosamente.")

    print("="*60 + "\n")
