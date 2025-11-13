"""
Helper para configurar el entorno de los notebooks.
Importar al inicio de cada notebook para garantizar que los imports funcionen.
"""
import sys
from pathlib import Path

def setup_notebook_path():
    """
    Configura el path para que los imports funcionen desde cualquier notebook.
    Detecta automáticamente la raíz del proyecto.
    """
    # Obtener el directorio actual del notebook
    notebook_dir = Path.cwd()

    # Buscar la raíz del proyecto (donde está utils/)
    current = notebook_dir
    max_levels = 5

    for _ in range(max_levels):
        if (current / 'utils').exists():
            # Encontramos la raíz
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        current = current.parent

    # Si no encontramos, intentar con ../..
    fallback = notebook_dir / '..' / '..'
    sys.path.insert(0, str(fallback.resolve()))
    return fallback.resolve()

# Configurar automáticamente al importar
project_root = setup_notebook_path()
print(f"✅ Proyecto configurado. Raíz: {project_root}")
