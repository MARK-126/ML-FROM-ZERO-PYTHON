# 🎓 Machine Learning desde CERO con Python

Un curso completo e interactivo de Machine Learning implementado DESDE CERO usando únicamente Python, NumPy y Jupyter Notebooks.

## 📚 Descripción

Este proyecto es un curso práctico y teórico de Machine Learning donde implementarás todos los algoritmos fundamentales desde cero, sin usar librerías de alto nivel como scikit-learn (excepto para comparaciones). Aprenderás la matemática y la implementación real detrás de cada algoritmo.

## ✨ Características

- 📝 **Notebooks interactivos** con explicaciones conceptuales detalladas
- 💻 **Implementaciones desde cero** de todos los algoritmos usando solo NumPy
- ✅ **Tests automatizados** para verificar tus ejercicios
- 📊 **Visualizaciones** con Matplotlib y Seaborn
- 🎯 **Ejercicios prácticos** con soluciones verificables
- 🧪 **Datasets sintéticos** para experimentación

## 🗂️ Estructura del Proyecto

```
ML-FROM-ZERO-PYTHON/
├── notebooks/
│   ├── fundamentos/
│   │   ├── 01_introduccion_python_ml.ipynb
│   │   ├── 02_numpy_arrays.ipynb
│   │   ├── 03_pandas_datos.ipynb
│   │   ├── 04_matplotlib_visualizacion.ipynb
│   │   └── 05_fundamentos_ml.ipynb
│   ├── algoritmos-supervisados/
│   │   ├── 01_regresion_linear.ipynb
│   │   ├── 02_regresion_logistica.ipynb
│   │   └── 03_knn.ipynb
│   ├── algoritmos-no-supervisados/
│   │   └── 01_kmeans.ipynb
│   └── deep-learning/
│       └── 01_redes_neuronales.ipynb
├── utils/
│   ├── __init__.py
│   ├── test_utils.py
│   └── plot_utils.py
├── data/
├── requirements.txt
└── README.md
```

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clona el repositorio**
```bash
git clone <URL_DEL_REPO>
cd ML-FROM-ZERO-PYTHON
```

2. **Crea un entorno virtual (recomendado)**
```bash
python -m venv venv

# Activar en Windows
venv\Scripts\activate

# Activar en Linux/Mac
source venv/bin/activate
```

3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

4. **Instala el paquete utils en modo desarrollo (RECOMENDADO)**
```bash
pip install -e .
```
Esto permite importar `utils` desde cualquier notebook sin problemas de rutas.

5. **Inicia Jupyter Notebook**
```bash
jupyter notebook
```

### ⚠️ Importante: Configuración de Rutas

**Si NO instalaste el paquete con `pip install -e .`**, necesitas configurar las rutas manualmente:

**Opción A (Recomendada)**: Al inicio de cada notebook, usa:
```python
# En lugar de sys.path.append('../..')
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
```

**Opción B**: Usa el helper incluido:
```python
from notebook_setup import setup_notebook_path
setup_notebook_path()
```

**Opción C**: Siempre ejecuta Jupyter desde la raíz del proyecto:
```bash
cd ML-FROM-ZERO-PYTHON
jupyter notebook
# Abre notebooks desde la interfaz web
```

## 📖 Contenido del Curso

### Módulo 1: Fundamentos

#### 01 - Introducción a Python para ML
- Listas, diccionarios y estructuras de datos
- Funciones y programación orientada a objetos
- List comprehension
- Lambda functions

#### 02 - NumPy
- Creación y manipulación de arrays
- Indexación y slicing avanzado
- Operaciones vectorizadas
- Broadcasting
- Álgebra lineal

#### 03 - Pandas
- Series y DataFrames
- Filtrado y selección de datos
- Manejo de datos faltantes
- GroupBy y agregaciones
- Merge y concatenación

#### 04 - Matplotlib
- Gráficos básicos (líneas, scatter, histogramas)
- Subplots
- Personalización de visualizaciones
- Seaborn para gráficos estadísticos

#### 05 - Fundamentos de Machine Learning
- Tipos de ML: Supervisado, No Supervisado, Refuerzo
- Train/Test Split
- Overfitting vs Underfitting
- Métricas de evaluación
- Cross-validation

### Módulo 2: Algoritmos Supervisados

#### 06 - Regresión Linear
**Implementación DESDE CERO**
- Teoría matemática completa
- Gradient Descent
- Función de costo (MSE)
- Regresión con múltiples variables
- Normalización de features
- R² Score

**Lo que aprenderás:**
```python
class RegresionLinear:
    def fit(self, X, y):
        # Implementación con Gradient Descent

    def predict(self, X):
        # Predicciones
```

#### 07 - Regresión Logística
**Implementación DESDE CERO**
- Función sigmoide
- Binary Cross-Entropy Loss
- Gradient Descent para clasificación
- Métricas: Accuracy, Precision, Recall, F1
- Matriz de confusión

#### 08 - K-Nearest Neighbors (KNN)
**Implementación DESDE CERO**
- Algoritmo KNN para clasificación
- Distancia euclidiana
- Elegir K óptimo
- KNN para regresión

### Módulo 3: Algoritmos No Supervisados

#### 09 - K-Means Clustering
**Implementación DESDE CERO**
- Algoritmo de K-Means
- Inicialización de centroides
- Método del codo (Elbow Method)
- Visualización de clusters
- Convergencia

### Módulo 4: Deep Learning

#### 10 - Redes Neuronales
**Implementación DESDE CERO**
- Perceptrón multicapa
- Forward Propagation
- Backpropagation
- Funciones de activación
- Clasificación no lineal

## 🎯 Cómo Usar Este Curso

### Para Estudiantes

1. **Sigue el orden**: Los notebooks están numerados y diseñados para seguirse secuencialmente
2. **Lee la teoría**: Cada notebook comienza con explicaciones conceptuales
3. **Ejecuta los ejemplos**: Corre todas las celdas para ver los resultados
4. **Haz los ejercicios**: Completa los ejercicios marcados con 🎯
5. **Verifica tus respuestas**: Usa los tests automatizados incluidos

### Ejemplo de Uso de Tests

```python
# Tu código
mi_array = np.array([1, 2, 3, 4, 5])
mi_promedio = np.mean(mi_array)

# Test automático
result = check_answer(mi_promedio, 3.0, answer_type="value")
print(result)  # ✅ TEST PASADO: Respuesta correcta: 3.0
```

### Para Instructores

- Notebooks listos para usar en clases
- Tests automatizados para calificar ejercicios
- Visualizaciones claras para explicar conceptos
- Código comentado y bien documentado

## 🧪 Tests Automatizados

El proyecto incluye utilidades para verificar automáticamente las respuestas de los estudiantes:

### Funciones Disponibles

```python
from utils.test_utils import (
    check_answer,      # Verificar valores, arrays, strings
    check_array,       # Verificar arrays de NumPy
    check_function,    # Verificar funciones con test cases
    check_model,       # Verificar modelos de ML
    print_test_summary # Imprimir resumen de múltiples tests
)
```

### Ejemplo

```python
# Verificar una respuesta numérica
check_answer(student_answer, expected_answer, tolerance=1e-5)

# Verificar un array
check_array(student_array, expected_array, tolerance=1e-5)

# Verificar una función
test_cases = [
    ([1, 2, 3], 2.0),  # (input, expected_output)
    ([10, 20], 15.0)
]
check_function(student_function, test_cases)
```

## 📊 Utilidades de Visualización

```python
from utils.plot_utils import (
    plot_decision_boundary,   # Fronteras de decisión
    plot_confusion_matrix,    # Matriz de confusión
    plot_learning_curve,      # Curvas de aprendizaje
    plot_regression_results,  # Resultados de regresión
    plot_clusters            # Visualizar clusters
)
```

## 🎓 Algoritmos Implementados Desde Cero

| Algoritmo | Tipo | Notebook | Técnicas |
|-----------|------|----------|----------|
| **Regresión Linear** | Supervisado - Regresión | 06 | Gradient Descent, MSE |
| **Regresión Logística** | Supervisado - Clasificación | 07 | Sigmoide, Binary Cross-Entropy |
| **K-Nearest Neighbors** | Supervisado - Clasificación/Regresión | 08 | Distancia Euclidiana |
| **K-Means** | No Supervisado - Clustering | 09 | Método del Codo |
| **Red Neuronal** | Supervisado - Clasificación | 10 | Backpropagation, Forward Prop |

## 🏗️ Conceptos Implementados

- ✅ Gradient Descent
- ✅ Backpropagation
- ✅ Cost Functions (MSE, Binary Cross-Entropy)
- ✅ Activaciones (Sigmoide)
- ✅ Normalización de datos
- ✅ Train/Test Split
- ✅ Métricas de evaluación
- ✅ Matrices de confusión

## 💡 Ejercicios Incluidos

Cada notebook incluye:
- **Ejercicios guiados** con instrucciones claras
- **Ejercicios de implementación** para crear tus propias versiones
- **Ejercicios integradores** que combinan múltiples conceptos
- **Tests automatizados** para verificación inmediata

## 🤝 Contribuciones

Este es un proyecto educativo. Si encuentras errores o tienes sugerencias:

1. Abre un issue describiendo el problema/sugerencia
2. Si quieres contribuir código, haz un fork y pull request

## 📝 Licencia

Este proyecto es de código abierto y está disponible para uso educativo.

## 🎯 Objetivos de Aprendizaje

Al completar este curso, serás capaz de:

1. ✅ Implementar algoritmos de ML desde cero en Python
2. ✅ Entender la matemática detrás de cada algoritmo
3. ✅ Usar NumPy para operaciones matriciales eficientes
4. ✅ Visualizar datos y resultados de modelos
5. ✅ Evaluar modelos con métricas apropiadas
6. ✅ Detectar y prevenir overfitting
7. ✅ Implementar Gradient Descent y Backpropagation
8. ✅ Trabajar con datos reales usando Pandas

## 📚 Recursos Adicionales

### Libros Recomendados
- "Hands-On Machine Learning" - Aurélien Géron
- "Pattern Recognition and Machine Learning" - Christopher Bishop
- "Deep Learning" - Ian Goodfellow

### Cursos Online
- Andrew Ng - Machine Learning (Coursera)
- Fast.ai - Practical Deep Learning

### Documentación
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## 🚀 Próximos Pasos

Después de completar este curso, considera:

1. **Implementar más algoritmos**: SVM, Random Forests, Gradient Boosting
2. **Proyectos reales**: Kaggle competitions
3. **Deep Learning avanzado**: CNNs, RNNs, Transformers
4. **Frameworks modernos**: TensorFlow, PyTorch

## ⭐ Características Destacadas

- 🎓 **Pedagógico**: Diseñado específicamente para aprendizaje
- 🔬 **Científico**: Implementaciones matemáticamente correctas
- 💪 **Práctico**: Ejercicios y proyectos aplicados
- ✅ **Verificable**: Tests automatizados incluidos
- 📊 **Visual**: Gráficos y visualizaciones en cada paso
- 🐍 **Pythónico**: Código limpio y bien documentado

## ⚠️ Problemas Conocidos y Limitaciones

### Rutas de Importación
**Problema**: Los notebooks usan `sys.path.append('../..')` que es frágil.

**Solución**: Instala el proyecto con `pip install -e .` como se indica en la sección de instalación.

**Alternativa**: Usa el helper `notebook_setup.py` o siempre ejecuta Jupyter desde la raíz del proyecto.

### Uso de Scikit-learn
Aunque implementamos los algoritmos desde cero, usamos `scikit-learn` para:
- Generar datasets sintéticos (`make_classification`, `make_moons`, etc.)
- Comparaciones y validación (opcional)

Esto es intencional y no afecta el aprendizaje de las implementaciones desde cero.

### Rendimiento
Las implementaciones priorizan **claridad y educación** sobre rendimiento. Las versiones de producción (scikit-learn, TensorFlow) están altamente optimizadas en C/C++.

### Compatibilidad
- Probado en Python 3.8+
- Puede haber warnings de NumPy/Pandas con versiones muy nuevas
- Se recomienda usar las versiones especificadas en `requirements.txt`

## 🙏 Agradecimientos

Este proyecto fue creado con el objetivo de hacer Machine Learning accesible para todos, implementando cada algoritmo desde sus fundamentos matemáticos hasta el código Python.

---

**¡Comienza tu viaje en Machine Learning hoy!** 🚀

Para cualquier pregunta o problema, abre un issue en el repositorio.

**Happy Learning!** 🎓✨
