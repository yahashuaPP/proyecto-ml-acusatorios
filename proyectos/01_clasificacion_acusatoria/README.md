# Clasificación de Contratos: Sistema Acusatorio 

Este proyecto implementa un sistema de Procesamiento de Lenguaje Natural (NLP) para la categorización automatizada de documentos legales, estructurado bajo los pilares fundamentales del aprendizaje automático.

##  Framework Teórico

Para garantizar la precisión y robustez del sistema, el desarrollo se divide en tres componentes esenciales:

### 1. Representación
Es el espacio de hipótesis donde el modelo "entiende" los datos.
* **Modelo Base:** BETO (Spanish BERT).
* **Embeddings:** Transformación del lenguaje jurídico a vectores densos de alta dimensionalidad.
* **Espacio de Características:** Uso de mecanismos de atención para capturar dependencias semánticas en textos legales complejos.

### 2. Optimización
El proceso de búsqueda de los parámetros óptimos para el modelo.
* **Estrategia:** Fine-tuning de los pesos del transformador mediante el optimizador AdamW.
* **AutoML:** Implementación de **Optuna** para la búsqueda bayesiana de hiperparámetros, optimizando la arquitectura de forma eficiente y sistemática.

### 3. Evaluación
La métrica de éxito para validar la capacidad de generalización del modelo.
* **Métricas Principales:** F1-Score (Macro y Weighted) para manejar el desbalance de clases en los contratos.
* **Análisis de Error:** Matriz de confusión para identificar solapamientos entre categorías legales y asegurar la fiabilidad del clasificador.

---

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python
* **Librerías:** Transformers (Hugging Face), Scikit-learn, Pandas.
* **Optimización:** Optuna.

## 📊 Archivos
* `proyecto_ml_kapak_final_corregido.ipynb`: Notebook con el flujo completo de ingeniería de datos, entrenamiento y validación.
* https://drive.google.com/file/d/1YqDCZVOYT49ZBWDeamwzJP-fwgYo_Vqk/view?usp=sharing
