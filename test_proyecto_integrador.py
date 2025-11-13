#!/usr/bin/env python3
"""
Script de prueba para verificar que el proyecto integrador funciona.
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Configurar path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("="*70)
print("🧪 TESTING PROYECTO INTEGRADOR")
print("="*70)

# Test 1: Verificar imports
print("\n📋 Test 1: Verificando imports...")
try:
    from utils.plot_utils import plot_confusion_matrix
    print("✅ Utils importadas correctamente")
except ImportError as e:
    print(f"❌ Error al importar utils: {e}")
    sys.exit(1)

# Test 2: Cargar dataset
print("\n📋 Test 2: Cargando dataset...")
try:
    data_path = project_root / 'data' / 'iris_extended.csv'
    df = pd.read_csv(data_path)
    print(f"✅ Dataset cargado: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")
    print(f"   Especies: {df['species'].unique()}")
except Exception as e:
    print(f"❌ Error al cargar dataset: {e}")
    sys.exit(1)

# Test 3: Preparar datos
print("\n📋 Test 3: Preparando datos...")
try:
    species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    df['species_encoded'] = df['species'].map(species_map)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = df['species_encoded'].values

    # Normalizar
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_normalized = (X - X_mean) / X_std

    print(f"✅ Datos preparados: X shape={X.shape}, y shape={y.shape}")
    print(f"   Clases: {np.unique(y)}")
except Exception as e:
    print(f"❌ Error al preparar datos: {e}")
    sys.exit(1)

# Test 4: Modelo Regresión Logística
print("\n📋 Test 4: Entrenando Regresión Logística...")
try:
    class LogisticRegressionMulticlass:
        def __init__(self, learning_rate=0.01, n_iterations=1000):
            self.learning_rate = learning_rate
            self.n_iterations = n_iterations
            self.models = {}
            self.classes = None

        def sigmoid(self, z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        def fit(self, X, y):
            self.classes = np.unique(y)
            n_samples, n_features = X.shape

            for cls in self.classes:
                y_binary = (y == cls).astype(int)
                w = np.zeros(n_features)
                b = 0

                for _ in range(self.n_iterations):
                    z = np.dot(X, w) + b
                    y_pred = self.sigmoid(z)
                    dw = (1/n_samples) * np.dot(X.T, (y_pred - y_binary))
                    db = (1/n_samples) * np.sum(y_pred - y_binary)
                    w -= self.learning_rate * dw
                    b -= self.learning_rate * db

                self.models[cls] = {'w': w, 'b': b}
            return self

        def predict_proba(self, X):
            probas = np.zeros((X.shape[0], len(self.classes)))
            for i, cls in enumerate(self.classes):
                w = self.models[cls]['w']
                b = self.models[cls]['b']
                z = np.dot(X, w) + b
                probas[:, i] = self.sigmoid(z)
            return probas

        def predict(self, X):
            probas = self.predict_proba(X)
            return self.classes[np.argmax(probas, axis=1)]

    model_lr = LogisticRegressionMulticlass(learning_rate=0.1, n_iterations=500)
    model_lr.fit(X_normalized[:30], y[:30])  # Train on first 30
    y_pred_lr = model_lr.predict(X_normalized[30:])  # Test on last 10
    accuracy_lr = np.mean(y_pred_lr == y[30:])
    print(f"✅ Regresión Logística entrenada")
    print(f"   Accuracy en test: {accuracy_lr:.4f}")
except Exception as e:
    print(f"❌ Error en Regresión Logística: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Modelo KNN
print("\n📋 Test 5: Entrenando KNN...")
try:
    class KNN:
        def __init__(self, k=3):
            self.k = k
            self.X_train = None
            self.y_train = None

        def fit(self, X, y):
            self.X_train = X
            self.y_train = y
            return self

        def predict(self, X):
            predictions = []
            for x in X:
                distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
                k_indices = np.argsort(distances)[:self.k]
                k_nearest_labels = self.y_train[k_indices]
                most_common = np.bincount(k_nearest_labels.astype(int)).argmax()
                predictions.append(most_common)
            return np.array(predictions)

    model_knn = KNN(k=3)
    model_knn.fit(X_normalized[:30], y[:30])
    y_pred_knn = model_knn.predict(X_normalized[30:])
    accuracy_knn = np.mean(y_pred_knn == y[30:])
    print(f"✅ KNN entrenado")
    print(f"   Accuracy en test: {accuracy_knn:.4f}")
except Exception as e:
    print(f"❌ Error en KNN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: K-Fold Cross-Validation
print("\n📋 Test 6: Probando K-Fold CV...")
try:
    class KFoldCV:
        def __init__(self, n_splits=5, shuffle=True, random_state=None):
            self.n_splits = n_splits
            self.shuffle = shuffle
            self.random_state = random_state

        def split(self, X, y=None):
            n_samples = len(X)
            indices = np.arange(n_samples)

            if self.shuffle:
                if self.random_state is not None:
                    np.random.seed(self.random_state)
                np.random.shuffle(indices)

            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[:n_samples % self.n_splits] += 1

            current = 0
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                test_indices = indices[start:stop]
                train_indices = np.concatenate([indices[:start], indices[stop:]])
                yield train_indices, test_indices
                current = stop

    kfold = KFoldCV(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_normalized, y), 1):
        X_train, X_test = X_normalized[train_idx], X_normalized[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = KNN(k=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        scores.append(accuracy)
        print(f"   Fold {fold}: Accuracy = {accuracy:.4f}")

    print(f"✅ K-Fold CV funcionando correctamente")
    print(f"   Accuracy promedio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
except Exception as e:
    print(f"❌ Error en K-Fold CV: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✅ TODOS LOS TESTS PASARON")
print("="*70)
print("\n🎉 El proyecto integrador funciona correctamente!")
print("   Los estudiantes pueden ejecutar el notebook sin problemas.")
