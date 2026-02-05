
**Instrucciones:**

1.  En VS Code, crea un nuevo archivo llamado `README.md` (si no existe ya).
2.  Copia y pega **todo** el contenido de abajo.
3.  Guárdalo y haz un `git add .`, `git commit` y `git push` para subirlo.

-----

````markdown
# Identificador de Razas Caninas con IA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

***Aplicación web interactiva basada en Inteligencia Artificial capaz de identificar **120 razas de perros** a partir de imágenes. Utiliza una Red Neuronal Convolucional (CNN) optimizada para ofrecer predicciones rápidas y precisas, proporcionando además recomendaciones de cuidado y reproducción de audio para accesibilidad.

🔗 **[Ver Demo en Vivo](https://share.streamlit.io/)** *(Pega aquí tu link de Streamlit cuando lo tengas)*

---

## Características Principales

*  **Detección en Tiempo Real:** Usa la cámara de tu dispositivo o sube una imagen desde tu galería.
*  **Clasificación Inteligente:** Identifica entre 120 razas distintas utilizando el dataset de Stanford.
*  **Análisis Detallado:** Muestra las **Top 5 razas más probables** con un gráfico de confianza.
*  **Filtro de Seguridad:** Detecta si la imagen subida no es un perro (umbral de confianza < 40%).
*  **Comparación Visual:** Muestra una foto de referencia de la raza detectada junto a tu foto para validar el resultado.
*  **Audio-Guía:** Genera recomendaciones de cuidado habladas (Text-to-Speech) usando `gTTS`.
*  **Información de Cuidado:** Proporciona datos sobre temperamento, ejercicio y cuidado del pelaje.

---

##  Stack Tecnológico

* **Lenguaje:** Python 3
* **Deep Learning:** TensorFlow / Keras
* **Modelo Base:** MobileNetV2 (Pre-entrenado en ImageNet)
* **Frontend / Despliegue:** Streamlit
* **Procesamiento de Imágenes:** Pillow (PIL)
* **Audio:** gTTS (Google Text-to-Speech)
* **Datos:** Pandas, NumPy

---

##  Arquitectura del Modelo

El núcleo del proyecto es una **Red Neuronal Convolucional (CNN)** basada en la arquitectura **MobileNetV2**.

### Estrategia de Entrenamiento
1.  **Transfer Learning:** Utilizamos MobileNetV2 como extractor de características (congelado), aprovechando su pre-entrenamiento en ImageNet. Añadimos una cabecera personalizada (`GlobalAveragePooling` + `Dropout` + `Dense Softmax`) para nuestras 120 clases.
2.  **Fine-Tuning (Afinamiento):** Descongelamos las últimas 100 capas del modelo base y re-entrenamos con una tasa de aprendizaje muy baja (`1e-5`) para adaptar los filtros a detalles específicos de las razas caninas.
3.  **Resultados:** Se logró una precisión (accuracy) de validación superior al **81%**.

---

##  Estructura del Proyecto

```text
dog_id_project/
├── app.py                         # Código principal de la aplicación Streamlit
├── dog_breed_classifier_v2.keras  # Modelo entrenado (MobileNetV2 + Fine Tuning)
├── recommendations.json           # Base de datos de cuidados y descripciones
├── requirements.txt               # Lista de dependencias para instalación
├── breed_images/                  # Carpeta con 120 imágenes de referencia
│   ├── n02085620-Chihuahua.jpg
│   └── ...
└── README.md                      # Documentación del proyecto
````

-----

##  Instalación y Uso Local

Sigue estos pasos para ejecutar el proyecto en tu computadora:

1.  **Clonar el repositorio:**

    ```bash
    git clone [https://github.com/TU_USUARIO/dog_id_project.git](https://github.com/TU_USUARIO/dog_id_project.git)
    cd dog_id_project
    ```

2.  **Crear un entorno virtual (Opcional pero recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar la aplicación:**

    ```bash
    python -m streamlit run app.py
    ```

5.  **Abrir en el navegador:**
    La app estará disponible en `http://localhost:8501`.

-----

##  Dataset

El modelo fue entrenado utilizando el **[Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)**.

  * **Total de imágenes:** 20,580
  * **Clases:** 120 razas
  * **Fuente:** ImageNet

-----

## 👤 Autor


Desarrollado por **Estudiantes de la Universidad Privada Antenor Orrego - Curso : Inteligencia Artificial Principios y Tecnicas** como parte del curso de Inteligencia Artificial.

-----

```
```
