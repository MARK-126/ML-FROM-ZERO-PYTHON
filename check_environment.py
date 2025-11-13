#!/usr/bin/env python3
"""
Verifica que el entorno esté configurado correctamente.
Ejecuta esto antes de empezar: python check_environment.py
"""

import sys
import importlib
from pathlib import Path


def check_python_version():
    """Verifica versión de Python."""
    print("\n📋 Verificando versión de Python...")
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ requerido")
        print(f"   Tienes: Python {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_packages():
    """Verifica paquetes instalados."""
    print("\n📋 Verificando paquetes instalados...")

    required = {
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'matplotlib': '3.7.0',
        'seaborn': '0.12.0',
        'scipy': '1.10.0',
        'sklearn': '1.3.0',
        'jupyter': '1.0.0',
    }

    missing = []

    for pkg, min_version in required.items():
        pkg_import = pkg if pkg != 'sklearn' else 'sklearn'
        try:
            mod = importlib.import_module(pkg_import)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {pkg:15} v{version}")
        except ImportError:
            print(f"❌ {pkg:15} NO ENCONTRADO")
            missing.append(pkg)

    if missing:
        print(f"\n⚠️  Faltan {len(missing)} paquete(s)")
        print(f"   Ejecuta: pip install -r requirements.txt")

    return len(missing) == 0


def check_utils_import():
    """Verifica que utils/ se puede importar."""
    print("\n📋 Verificando módulo utils...")

    # Intentar agregar al path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from utils.test_utils import check_answer
        from utils.plot_utils import plot_decision_boundary
        print("✅ Módulo utils importable correctamente")
        return True
    except ImportError as e:
        print("❌ Módulo utils NO importable")
        print(f"   Error: {e}")
        print("\n💡 Solución: Ejecuta 'pip install -e .' desde la raíz del proyecto")
        return False


def check_directory_structure():
    """Verifica que la estructura de directorios sea correcta."""
    print("\n📋 Verificando estructura de directorios...")

    project_root = Path(__file__).parent
    required_dirs = [
        'notebooks/fundamentos',
        'notebooks/algoritmos-supervisados',
        'notebooks/algoritmos-no-supervisados',
        'notebooks/deep-learning',
        'notebooks/evaluacion',
        'utils',
        'data',
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - NO ENCONTRADO")
            all_ok = False

    return all_ok


def check_notebooks_exist():
    """Verifica que los notebooks principales existan."""
    print("\n📋 Verificando notebooks principales...")

    project_root = Path(__file__).parent
    required_notebooks = [
        'notebooks/fundamentos/01_introduccion_python_ml.ipynb',
        'notebooks/fundamentos/02_numpy_arrays.ipynb',
        'notebooks/algoritmos-supervisados/01_regresion_linear.ipynb',
        'notebooks/evaluacion/01_metricas_evaluacion.ipynb',
    ]

    all_ok = True
    for nb_path in required_notebooks:
        full_path = project_root / nb_path
        if full_path.exists():
            print(f"✅ {nb_path.split('/')[-1]}")
        else:
            print(f"❌ {nb_path.split('/')[-1]} - NO ENCONTRADO")
            all_ok = False

    return all_ok


def main():
    print("=" * 70)
    print("🔍 VERIFICACIÓN DE ENTORNO - ML FROM ZERO")
    print("=" * 70)

    checks = [
        ("Python version", check_python_version),
        ("Paquetes instalados", check_packages),
        ("Estructura de directorios", check_directory_structure),
        ("Notebooks principales", check_notebooks_exist),
        ("Módulo utils", check_utils_import),
    ]

    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error en verificación '{name}': {e}")
            results.append(False)

    print("\n" + "=" * 70)
    if all(results):
        print("🎉 ¡TODO CONFIGURADO CORRECTAMENTE!")
        print("\n📚 Próximos pasos:")
        print("   1. Inicia Jupyter: jupyter notebook")
        print("   2. Abre: notebooks/fundamentos/01_introduccion_python_ml.ipynb")
        print("   3. ¡Comienza a aprender!")
    else:
        print("⚠️  ALGUNOS PROBLEMAS ENCONTRADOS")
        print("\n🔧 Para resolver:")
        print("   1. Instala dependencias: pip install -r requirements.txt")
        print("   2. Instala el proyecto: pip install -e .")
        print("   3. Ejecuta nuevamente: python check_environment.py")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
