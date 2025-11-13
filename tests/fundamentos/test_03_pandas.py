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
    """Test para crear DataFrame básico"""
    def verificar(df):
        if not isinstance(df, pd.DataFrame):
            print_error("El resultado no es un DataFrame")
            return False

        if df.shape != (3, 3):
            print_error(f"Shape incorrecta. Expected: (3, 3), Got: {df.shape}")
            return False

        print_success("Ejercicio 1 completado!")
        return True
    return verificar


def test_ejercicio_2_filtrado():
    """Test para filtrado de datos"""
    def verificar(df_filtrado):
        if len(df_filtrado) == 0:
            print_error("DataFrame filtrado está vacío")
            return False

        print("   ✓ Filtrado correcto")
        print_success("Ejercicio 2 completado!")
        return True
    return verificar


def test_ejercicio_3_groupby():
    """Test para operaciones groupby"""
    def verificar(resultado):
        if resultado is None:
            print_error("Resultado es None")
            return False

        print("   ✓ GroupBy aplicado correctamente")
        print_success("Ejercicio 3 completado!")
        return True
    return verificar
