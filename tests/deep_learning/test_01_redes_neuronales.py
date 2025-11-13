"""
Tests para el notebook 01_redes_neuronales.ipynb

Este archivo contiene tests automáticos para verificar las implementaciones
de los ejercicios GRADED del notebook de Redes Neuronales.
"""

import numpy as np
from utils.testing_utils import print_success, print_error, print_info


def test_ejercicio_1_forward():
    """
    Test para la función de forward propagation en redes neuronales.

    Retorna
    -------
    function
        Función verificadora que acepta la función implementada por el estudiante
    """
    def verificar(forward_func):
        """
        Verifica la implementación de forward propagation.

        Parámetros
        ----------
        forward_func : function
            Función neural_network_forward implementada por el estudiante

        Retorna
        -------
        bool
            True si todos los tests pasan, False en caso contrario
        """
        all_passed = True

        # ==========================================
        # Test 1: Verificar dimensiones de salida
        # ==========================================
        try:
            np.random.seed(42)
            X = np.random.randn(5, 3)  # 5 ejemplos, 3 features
            W1 = np.random.randn(3, 4) * 0.01  # 4 neuronas ocultas
            b1 = np.zeros((1, 4))
            W2 = np.random.randn(4, 1) * 0.01  # 1 neurona de salida
            b2 = np.zeros((1, 1))

            Z1, A1, Z2, A2 = forward_func(X, W1, b1, W2, b2)

            # Verificar dimensiones
            assert Z1.shape == (5, 4), f"Z1 shape incorrecto: {Z1.shape}, esperado: (5, 4)"
            assert A1.shape == (5, 4), f"A1 shape incorrecto: {A1.shape}, esperado: (5, 4)"
            assert Z2.shape == (5, 1), f"Z2 shape incorrecto: {Z2.shape}, esperado: (5, 1)"
            assert A2.shape == (5, 1), f"A2 shape incorrecto: {A2.shape}, esperado: (5, 1)"

            print_success("Test 1 aprobado: Dimensiones correctas")
        except AssertionError as e:
            print_error(f"Test 1 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 1 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 2: Verificar valores de Z1
        # ==========================================
        try:
            np.random.seed(123)
            X = np.array([[1.0, 2.0], [3.0, 4.0]])
            W1 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            b1 = np.array([[0.1, 0.2, 0.3]])
            W2 = np.array([[0.7], [0.8], [0.9]])
            b2 = np.array([[0.4]])

            Z1_expected = X @ W1 + b1
            Z1, _, _, _ = forward_func(X, W1, b1, W2, b2)

            assert np.allclose(Z1, Z1_expected, rtol=1e-5), \
                f"Z1 valores incorrectos. Esperado:\n{Z1_expected}\nObtenido:\n{Z1}"

            print_success("Test 2 aprobado: Valores de Z1 correctos")
        except AssertionError as e:
            print_error(f"Test 2 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 2 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 3: Verificar que A1 tiene sigmoid aplicado
        # ==========================================
        try:
            def sigmoid(z):
                return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

            Z1, A1, _, _ = forward_func(X, W1, b1, W2, b2)
            A1_expected = sigmoid(Z1)

            assert np.allclose(A1, A1_expected, rtol=1e-5), \
                f"A1 no tiene sigmoid correctamente aplicado"

            # Verificar que A1 está entre 0 y 1
            assert np.all((A1 >= 0) & (A1 <= 1)), "A1 debe estar entre 0 y 1 (sigmoid)"

            print_success("Test 3 aprobado: Valores de A1 correctos (sigmoid aplicado)")
        except AssertionError as e:
            print_error(f"Test 3 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 3 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 4: Verificar valores de Z2
        # ==========================================
        try:
            Z1, A1, Z2, _ = forward_func(X, W1, b1, W2, b2)
            Z2_expected = A1 @ W2 + b2

            assert np.allclose(Z2, Z2_expected, rtol=1e-5), \
                f"Z2 valores incorrectos. Esperado:\n{Z2_expected}\nObtenido:\n{Z2}"

            print_success("Test 4 aprobado: Valores de Z2 correctos")
        except AssertionError as e:
            print_error(f"Test 4 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 4 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 5: Verificar que A2 tiene sigmoid aplicado
        # ==========================================
        try:
            Z1, A1, Z2, A2 = forward_func(X, W1, b1, W2, b2)
            A2_expected = sigmoid(Z2)

            assert np.allclose(A2, A2_expected, rtol=1e-5), \
                f"A2 no tiene sigmoid correctamente aplicado"

            # Verificar que A2 está entre 0 y 1 (probabilidades)
            assert np.all((A2 >= 0) & (A2 <= 1)), \
                "A2 debe estar entre 0 y 1 (probabilidades para clasificación binaria)"

            print_success("Test 5 aprobado: Valores de A2 correctos (probabilidades entre 0 y 1)")
        except AssertionError as e:
            print_error(f"Test 5 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 5 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 6: Test con diferentes dimensiones
        # ==========================================
        try:
            np.random.seed(999)
            X_test = np.random.randn(10, 5)  # 10 ejemplos, 5 features
            W1_test = np.random.randn(5, 8) * 0.01  # 8 neuronas ocultas
            b1_test = np.zeros((1, 8))
            W2_test = np.random.randn(8, 1) * 0.01
            b2_test = np.zeros((1, 1))

            Z1_t, A1_t, Z2_t, A2_t = forward_func(X_test, W1_test, b1_test, W2_test, b2_test)

            # Verificar cálculos completos
            Z1_expected = X_test @ W1_test + b1_test
            A1_expected = sigmoid(Z1_expected)
            Z2_expected = A1_expected @ W2_test + b2_test
            A2_expected = sigmoid(Z2_expected)

            assert np.allclose(Z1_t, Z1_expected, rtol=1e-5), "Z1 incorrecto para diferentes dimensiones"
            assert np.allclose(A1_t, A1_expected, rtol=1e-5), "A1 incorrecto para diferentes dimensiones"
            assert np.allclose(Z2_t, Z2_expected, rtol=1e-5), "Z2 incorrecto para diferentes dimensiones"
            assert np.allclose(A2_t, A2_expected, rtol=1e-5), "A2 incorrecto para diferentes dimensiones"

            print_success("Test 6 aprobado: Forward propagation completa funciona correctamente")
        except AssertionError as e:
            print_error(f"Test 6 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 6 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Resumen final
        # ==========================================
        if all_passed:
            print_success("\n🎉 ¡Todos los tests pasaron! Ejercicio 1 completado exitosamente.")
        else:
            print_error("\n❌ Algunos tests fallaron. Revisa tu implementación.")

        return all_passed

    return verificar


def test_ejercicio_2_gradients():
    """
    Test para la función de cálculo de gradientes (backpropagation).

    Retorna
    -------
    function
        Función verificadora que acepta la función implementada por el estudiante
    """
    def verificar(gradient_func):
        """
        Verifica la implementación de cálculo de gradientes.

        Parámetros
        ----------
        gradient_func : function
            Función compute_gradients implementada por el estudiante

        Retorna
        -------
        bool
            True si todos los tests pasan, False en caso contrario
        """
        all_passed = True

        # ==========================================
        # Test 1: Verificar dimensiones de gradientes
        # ==========================================
        try:
            np.random.seed(42)
            X = np.random.randn(5, 3)  # 5 ejemplos, 3 features
            y = np.array([0, 1, 1, 0, 1])
            A1 = np.random.rand(5, 4)  # 4 neuronas ocultas
            A2 = np.random.rand(5, 1)
            W2 = np.random.randn(4, 1) * 0.01

            dW1, db1 = gradient_func(X, y, A1, A2, W2)

            # Verificar dimensiones
            assert dW1.shape == (3, 4), f"dW1 shape incorrecto: {dW1.shape}, esperado: (3, 4)"
            assert db1.shape == (1, 4), f"db1 shape incorrecto: {db1.shape}, esperado: (1, 4)"

            print_success("Test 1 aprobado: Dimensiones de dW1 correctas")
        except AssertionError as e:
            print_error(f"Test 1 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 1 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 2: Verificar forma de db1
        # ==========================================
        try:
            assert db1.shape == (1, 4), f"db1 debe tener forma (1, 4) para broadcasting correcto"
            print_success("Test 2 aprobado: Dimensiones de db1 correctas")
        except AssertionError as e:
            print_error(f"Test 2 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 2 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 3: Verificar valores de dW1
        # ==========================================
        try:
            # Caso simple con valores conocidos
            X_test = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 3 ejemplos, 2 features
            y_test = np.array([0, 1, 0])
            A1_test = np.array([[0.5, 0.6, 0.7], [0.4, 0.5, 0.6], [0.3, 0.4, 0.5]])  # 3 neuronas ocultas
            A2_test = np.array([[0.3], [0.8], [0.4]])
            W2_test = np.array([[0.5], [0.6], [0.7]])

            # Calcular gradiente esperado manualmente
            m = 3
            y_test_reshaped = y_test.reshape(-1, 1)
            dZ2 = A2_test - y_test_reshaped
            dA1 = dZ2 @ W2_test.T
            dZ1 = dA1 * (A1_test * (1 - A1_test))  # Derivada de sigmoid
            dW1_expected = (1/m) * X_test.T @ dZ1
            db1_expected = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

            dW1_calc, db1_calc = gradient_func(X_test, y_test, A1_test, A2_test, W2_test)

            assert np.allclose(dW1_calc, dW1_expected, rtol=1e-5, atol=1e-8), \
                f"dW1 valores incorrectos.\nEsperado:\n{dW1_expected}\nObtenido:\n{dW1_calc}"

            print_success("Test 3 aprobado: Valores de dW1 correctos")
        except AssertionError as e:
            print_error(f"Test 3 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 3 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 4: Verificar valores de db1
        # ==========================================
        try:
            assert np.allclose(db1_calc, db1_expected, rtol=1e-5, atol=1e-8), \
                f"db1 valores incorrectos.\nEsperado:\n{db1_expected}\nObtenido:\n{db1_calc}"

            print_success("Test 4 aprobado: Valores de db1 correctos")
        except AssertionError as e:
            print_error(f"Test 4 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 4 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Test 5: Verificar con diferentes datos
        # ==========================================
        try:
            np.random.seed(777)
            X_rand = np.random.randn(8, 4)  # 8 ejemplos, 4 features
            y_rand = np.random.randint(0, 2, 8)  # Etiquetas binarias
            A1_rand = np.random.rand(8, 6)  # 6 neuronas ocultas
            A2_rand = np.random.rand(8, 1)
            W2_rand = np.random.randn(6, 1) * 0.01

            # Calcular gradientes esperados
            m_rand = 8
            y_rand_reshaped = y_rand.reshape(-1, 1)
            dZ2_rand = A2_rand - y_rand_reshaped
            dA1_rand = dZ2_rand @ W2_rand.T
            dZ1_rand = dA1_rand * (A1_rand * (1 - A1_rand))
            dW1_exp_rand = (1/m_rand) * X_rand.T @ dZ1_rand
            db1_exp_rand = (1/m_rand) * np.sum(dZ1_rand, axis=0, keepdims=True)

            dW1_res, db1_res = gradient_func(X_rand, y_rand, A1_rand, A2_rand, W2_rand)

            assert np.allclose(dW1_res, dW1_exp_rand, rtol=1e-5), \
                "dW1 incorrecto para datos aleatorios"
            assert np.allclose(db1_res, db1_exp_rand, rtol=1e-5), \
                "db1 incorrecto para datos aleatorios"

            print_success("Test 5 aprobado: Gradientes calculados correctamente para diferentes datos")
        except AssertionError as e:
            print_error(f"Test 5 fallido: {e}")
            all_passed = False
        except Exception as e:
            print_error(f"Test 5 error inesperado: {e}")
            all_passed = False

        # ==========================================
        # Resumen final
        # ==========================================
        if all_passed:
            print_success("\n🎉 ¡Todos los tests pasaron! Ejercicio 2 completado exitosamente.")
        else:
            print_error("\n❌ Algunos tests fallaron. Revisa tu implementación.")

        return all_passed

    return verificar
