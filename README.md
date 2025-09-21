# Red Neuronal Simple en PyTorch

Este proyecto implementa desde cero una clase de red neuronal secuencial en PyTorch, con funcionalidad de entrenamiento, evaluación y predicción usando el conjunto de datos MNIST (números escritos a mano).

---

## 📌 Objetivos Aprendidos

- Cómo construir una red neuronal personalizada con PyTorch (`torch.nn.Module`).
- Implementación manual del proceso de entrenamiento y validación:
  - Pérdida, optimización, métricas.
  - Lote por lote (batch training).
- Uso de GPU (si está disponible) con `torch.cuda`.
- Visualización del historial de entrenamiento con `matplotlib`.
- Evaluación del modelo sobre un conjunto de prueba.
- Predicciones y visualización de ejemplos usando `imshow`.

---

## 📊 Modelo Usado

La red neuronal consta de las siguientes capas:

- Capa Lineal: 784 → 100 (entrada)
- Función de activación ReLU
- Capa Lineal: 100 → 10 (salida, una clase por dígito)

---

## 📁 Estructura del Proyecto

- `MyModel`: Clase que define la arquitectura de la red.
- `MySequentialModel`: Clase de alto nivel con funciones `compile`, `fit`, `evaluate` y `predict`, al estilo Keras.
- `Accuracy`: Métrica personalizada de exactitud.
- Entrenamiento y evaluación sobre el conjunto **MNIST** descargado desde OpenML.

---

## 🧠 Ejemplo de Entrenamiento

Entrenamiento por 10 épocas con validación y visualización del historial de:
- `train_loss`, `val_loss`
- `train_acc`, `val_acc`

También se muestra un gráfico con las imágenes predichas y su clase estimada vs la real.

---

## 🚀 Ejecución

1. Clona este repositorio o copia el código en un archivo `.py`
2. Instala los requerimientos:

```bash
pip install -r requirements.txt


🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
