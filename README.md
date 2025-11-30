# Clasificación de Razas Caninas y Recomendaciones de Cuidado 🐶

![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?style=flat&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## 📖 Descripción
Este proyecto utiliza Inteligencia Artificial para identificar razas de perros a partir de imágenes y ofrecer recomendaciones de cuidado personalizadas. El sistema ayuda a adoptantes y dueños a entender mejor las necesidades de sus mascotas.

## 🧠 Modelo y Arquitectura
El núcleo del sistema se basa en **ResNet50** con Transfer Learning.
- **Dataset:** Stanford Dogs (~20k imágenes).
- **Modelo Final:** ResNet50 (Seleccionado por mejor rendimiento).
- **Técnicas:** Data Augmentation, Fine-Tuning.

## 📊 Resultados de Evaluación
| Métrica | Valor (ResNet50) |
|---------|------------------|
| Accuracy| 91%              |
| F1-Score| 0.91             |
| Recall  | 0.91             |

El modelo demostró una excelente generalización y baja tasa de confusión entre razas similares.

## 🚀 Demo en Vivo
Prueba la aplicación aquí: [Link a tu Streamlit](https://proyectorazasia-vhgt3pdzgscdkwmcgzsqys.streamlit.app/)

## 🛠 Instalación Local
1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/MauricioF68/proyecto_razas_ia](https://github.com/MauricioF68/proyecto_razas_ia)
