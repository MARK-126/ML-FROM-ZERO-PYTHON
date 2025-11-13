#!/usr/bin/env python3
"""
Script para actualizar las celdas de imports en todos los notebooks.
"""
import json
from pathlib import Path

# Template de configuración robusto
SETUP_TEMPLATE = """# ==========================================
# CONFIGURACIÓN DEL ENTORNO
# ==========================================
{imports}
import sys
from pathlib import Path

# Agregar el directorio raíz al path de manera robusta
project_root = Path.cwd().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verificar que las utilidades se pueden importar
try:
    {test_imports}
    print("✅ Entorno configurado correctamente")
    print(f"📁 Raíz del proyecto: {{project_root}}")
except ImportError as e:
    print("❌ Error al importar utilidades")
    print("\\n💡 Soluciones:")
    print("   1. Ejecuta 'pip install -e .' desde la raíz del proyecto")
    print("   2. O inicia Jupyter desde la raíz: cd ML-FROM-ZERO-PYTHON && jupyter notebook")
    print(f"\\n🔍 Error detallado: {{e}}")
    raise"""

notebooks_to_update = [
    # Algoritmos supervisados
    ("notebooks/algoritmos-supervisados/01_regresion_linear.ipynb",
     "import numpy as np\nimport matplotlib.pyplot as plt",
     "from utils.test_utils import check_answer, check_model, print_test_summary\n    from utils.plot_utils import plot_regression_results, plot_learning_curve"),

    ("notebooks/algoritmos-supervisados/02_regresion_logistica.ipynb",
     "import numpy as np\nimport matplotlib.pyplot as plt",
     "from utils.test_utils import check_answer\n    from utils.plot_utils import plot_decision_boundary"),

    ("notebooks/algoritmos-supervisados/03_knn.ipynb",
     "import numpy as np\nimport matplotlib.pyplot as plt\nfrom collections import Counter",
     "from utils.plot_utils import plot_decision_boundary"),

    # Algoritmos no supervisados
    ("notebooks/algoritmos-no-supervisados/01_kmeans.ipynb",
     "import numpy as np\nimport matplotlib.pyplot as plt",
     "from utils.plot_utils import plot_clusters"),

    # Deep Learning
    ("notebooks/deep-learning/01_redes_neuronales.ipynb",
     "import numpy as np\nimport matplotlib.pyplot as plt",
     "from utils.plot_utils import plot_decision_boundary, plot_learning_curve"),
]

def update_notebook(notebook_path, imports, test_imports):
    """Actualiza la primera celda de código de un notebook."""
    path = Path(notebook_path)
    if not path.exists():
        print(f"⚠️  No encontrado: {notebook_path}")
        return False

    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Encontrar la primera celda de código (después del markdown inicial)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and i > 0:  # Saltar la primera si es markdown
            # Actualizar la fuente de la celda
            new_source = SETUP_TEMPLATE.format(
                imports=imports,
                test_imports=test_imports
            )
            cell['source'] = new_source
            break

    # Guardar notebook actualizado
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✅ Actualizado: {notebook_path}")
    return True

if __name__ == "__main__":
    print("🔧 Actualizando imports en notebooks...\n")

    success_count = 0
    for notebook_path, imports, test_imports in notebooks_to_update:
        if update_notebook(notebook_path, imports, test_imports):
            success_count += 1

    print(f"\n🎉 {success_count}/{len(notebooks_to_update)} notebooks actualizados")
