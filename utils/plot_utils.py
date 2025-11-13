"""
Utilidades para visualizaciones en Machine Learning
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def plot_decision_boundary(X, y, model, title="Decision Boundary",
                          resolution=0.02, ax=None):
    """
    Visualiza la frontera de decisión de un modelo de clasificación.

    Parámetros:
    -----------
    X : np.ndarray
        Datos de entrada (debe tener 2 características)
    y : np.ndarray
        Etiquetas
    model : object
        Modelo con método predict
    title : str
        Título del gráfico
    resolution : float
        Resolución de la malla
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if X.shape[1] != 2:
        raise ValueError("Esta función solo funciona con datos de 2 dimensiones")

    # Configurar colores
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Crear malla
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predecir para toda la malla
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Crear el plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    # Plot de los puntos de datos
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                  alpha=0.8, c=colors[idx],
                  marker=markers[idx], label=f'Clase {cl}',
                  edgecolor='black', s=100)

    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    return ax


def plot_confusion_matrix(cm, classes=None, title='Matriz de Confusión',
                         normalize=False, cmap='Blues', ax=None):
    """
    Visualiza una matriz de confusión.

    Parámetros:
    -----------
    cm : np.ndarray
        Matriz de confusión
    classes : list
        Lista de nombres de clases
    title : str
        Título del gráfico
    normalize : bool
        Si True, normaliza la matriz
    cmap : str
        Mapa de colores
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    if classes is None:
        classes = [f'Clase {i}' for i in range(len(cm))]

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Etiqueta Real',
           xlabel='Etiqueta Predicha')

    # Rotar las etiquetas del eje x
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Añadir los valores en cada celda
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    return ax


def plot_learning_curve(train_scores, val_scores, train_sizes=None,
                       title='Curva de Aprendizaje', ax=None):
    """
    Visualiza la curva de aprendizaje de un modelo.

    Parámetros:
    -----------
    train_scores : list or np.ndarray
        Scores de entrenamiento
    val_scores : list or np.ndarray
        Scores de validación
    train_sizes : list or np.ndarray
        Tamaños del conjunto de entrenamiento (opcional)
    title : str
        Título del gráfico
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if train_sizes is None:
        train_sizes = np.arange(1, len(train_scores) + 1)

    ax.plot(train_sizes, train_scores, 'o-', color='r',
            label='Score de Entrenamiento', linewidth=2, markersize=8)
    ax.plot(train_sizes, val_scores, 'o-', color='g',
            label='Score de Validación', linewidth=2, markersize=8)

    ax.set_xlabel('Tamaño del conjunto de entrenamiento / Época')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    return ax


def plot_feature_importance(feature_names, importances, title='Importancia de Características',
                           top_n=None, ax=None):
    """
    Visualiza la importancia de las características.

    Parámetros:
    -----------
    feature_names : list
        Nombres de las características
    importances : np.ndarray
        Importancias de las características
    title : str
        Título del gráfico
    top_n : int
        Número de características más importantes a mostrar
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Ordenar por importancia
    indices = np.argsort(importances)[::-1]

    if top_n is not None:
        indices = indices[:top_n]

    ax.barh(range(len(indices)), importances[indices], color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importancia')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    return ax


def plot_regression_results(y_true, y_pred, title='Resultados de Regresión', ax=None):
    """
    Visualiza los resultados de un modelo de regresión.

    Parámetros:
    -----------
    y_true : np.ndarray
        Valores reales
    y_pred : np.ndarray
        Valores predichos
    title : str
        Título del gráfico
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='k')

    # Línea de predicción perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
            label='Predicción Perfecta')

    ax.set_xlabel('Valores Reales')
    ax.set_ylabel('Valores Predichos')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_clusters(X, labels, centers=None, title='Clustering', ax=None):
    """
    Visualiza los clusters.

    Parámetros:
    -----------
    X : np.ndarray
        Datos (debe tener 2 características)
    labels : np.ndarray
        Etiquetas de cluster
    centers : np.ndarray
        Centros de los clusters (opcional)
    title : str
        Título del gráfico
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if X.shape[1] != 2:
        raise ValueError("Esta función solo funciona con datos de 2 dimensiones")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Plot de los puntos
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis',
                        alpha=0.6, edgecolors='k', s=100)

    # Plot de los centros
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X',
                  s=300, edgecolors='k', linewidths=2, label='Centros')

    ax.set_xlabel('Característica 1')
    ax.set_ylabel('Característica 2')
    ax.set_title(title)
    if centers is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cluster')

    return ax


def plot_correlation_matrix(data, title='Matriz de Correlación', ax=None):
    """
    Visualiza una matriz de correlación.

    Parámetros:
    -----------
    data : pd.DataFrame or np.ndarray
        Datos
    title : str
        Título del gráfico
    ax : matplotlib.axes.Axes
        Eje donde dibujar (opcional)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if hasattr(data, 'corr'):
        corr = data.corr()
    else:
        corr = np.corrcoef(data.T)

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, ax=ax, cbar_kws={'label': 'Correlación'})

    ax.set_title(title)

    return ax
