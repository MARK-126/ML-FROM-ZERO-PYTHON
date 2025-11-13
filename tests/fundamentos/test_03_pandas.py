"""
Tests para Notebook 03: Pandas - Manipulación de Datos
========================================================
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.testing_utils import print_success, print_error


def test_ejercicio_1_crear_dataframe():
    """Test para crear DataFrame de productos"""
    def verificar(df):
        if df is None:
            print_error("El DataFrame es None")
            return False

        if not isinstance(df, pd.DataFrame):
            print_error("El resultado no es un DataFrame")
            return False

        if df.shape != (4, 4):
            print_error(f"Shape incorrecta. Expected: (4, 4), Got: {df.shape}")
            return False

        expected_columns = {'Nombre', 'Precio', 'Stock', 'Categoría'}
        actual_columns = set(df.columns)

        if not expected_columns.issubset(actual_columns):
            print_error(f"Columnas incorrectas. Expected: {expected_columns}, Got: {actual_columns}")
            return False

        print("   ✓ DataFrame con 4 filas y 4 columnas")
        print("   ✓ Columnas correctas: Nombre, Precio, Stock, Categoría")
        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_filtrado():
    """Test para filtrado de salarios altos"""
    def verificar(df_filtrado):
        if df_filtrado is None:
            print_error("El DataFrame filtrado es None")
            return False

        if not isinstance(df_filtrado, pd.DataFrame):
            print_error("El resultado no es un DataFrame")
            return False

        if len(df_filtrado) != 3:
            print_error(f"Número de filas incorrecto. Expected: 3, Got: {len(df_filtrado)}")
            print("   Hint: Filtra salarios >= 35000")
            return False

        # Verificar que todos los salarios sean >= 35000
        if not all(df_filtrado['Salario'] >= 35000):
            print_error("Algunos salarios son menores a 35000")
            return False

        print("   ✓ Filtrado correcto: 3 empleados con salario >= 35000")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar


def test_ejercicio_3_groupby():
    """Test para contar personas por ciudad"""
    def verificar(resultado):
        if resultado is None:
            print_error("Resultado es None")
            return False

        if not isinstance(resultado, pd.Series):
            print_error(f"El resultado debe ser una Series. Got: {type(resultado)}")
            return False

        # Verificar valores esperados
        expected = {'Madrid': 2, 'Barcelona': 2, 'Valencia': 1}

        all_correct = True
        for ciudad, count in expected.items():
            if ciudad not in resultado.index:
                print_error(f"Falta la ciudad '{ciudad}' en el resultado")
                all_correct = False
            elif resultado[ciudad] != count:
                print_error(f"Conteo incorrecto para {ciudad}. Expected: {count}, Got: {resultado[ciudad]}")
                all_correct = False

        if not all_correct:
            return False

        print("   ✓ Conteo correcto por ciudad:")
        print(f"     - Madrid: {resultado['Madrid']}")
        print(f"     - Barcelona: {resultado['Barcelona']}")
        print(f"     - Valencia: {resultado['Valencia']}")
        print_success("Ejercicio 3 completado!")
        return True
    return verificar
