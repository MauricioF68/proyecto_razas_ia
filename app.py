import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
from gtts import gTTS
import io
import os
import pandas as pd

# --- Configuración de la Página ---
# Usamos un ícono de perro 🐾 y forzamos el modo oscuro (dark)
st.set_page_config(
    page_title="UPAO: Analizador de Razas",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. Cargar el Modelo y los Datos ---

@st.cache_resource
def load_my_model():
    """Carga el modelo Keras entrenado."""
    try:
        model = tf.keras.models.load_model('dog_breed_classifier_v2_120_breeds.keras')
        return model
    except Exception as e:
        st.error(f"Error crítico al cargar el modelo: {e}")
        return None

@st.cache_data
def load_recommendations():
    """Carga el archivo JSON de recomendaciones."""
    try:
        with open('recommendations.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error crítico al cargar las recomendaciones: {e}")
        return None

model = load_my_model()
recommendations = load_recommendations()

# Obtenemos la lista de nombres de clases (los IDs) desde las claves del JSON
if recommendations:
    class_names = list(recommendations.keys())
else:
    class_names = []

# --- 2. Funciones Auxiliares ---

def process_image(pil_image):
    """Convierte una imagen PIL al formato que el modelo necesita."""
    img = np.array(pil_image)
    img_resized = tf.image.resize(img, (224, 224))
    img_expanded = tf.expand_dims(img_resized, axis=0)
    return img_expanded

def generate_audio(text):
    """Genera un archivo de audio en memoria a partir del texto."""
    try:
        tts = gTTS(text=text, lang='es', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error al generar el audio: {e}")
        return None

# --- 3. Función Principal de la App ---

def main():
    st.title("UPAO: Analizador de Razas Caninas")
    st.write("Identifica la raza de tu perro. Usa tu cámara o sube una foto para comenzar.")

    if model is None or recommendations is None:
        st.error("La aplicación no puede iniciar. Faltan archivos esenciales.")
        return

    # --- PESTAÑAS PARA LA ENTRADA ---
    tab1, tab2 = st.tabs(["📸 Usar Cámara", "⬆️ Subir Archivo"])
    img_file = None

    with tab1:
        st.info("Apunta la cámara de tu dispositivo al perro y toma una foto.")
        img_file_buffer = st.camera_input("Tomar foto", label_visibility="collapsed")
        if img_file_buffer:
            img_file = img_file_buffer

    with tab2:
        st.info("Sube una foto de tu perro desde tu galería (formatos: .jpg, .jpeg, .png).")
        img_file_buffer = st.file_uploader("Selecciona una foto", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if img_file_buffer:
            img_file = img_file_buffer

    # --- Lógica de Procesamiento (se ejecuta si hay una imagen) ---
    # ... (código anterior igual) ...

    # --- Lógica de Procesamiento ---
    if img_file is not None:
        pil_image = Image.open(img_file).convert('RGB')
        
        with st.spinner("Analizando... 🧠"):
            
            # Procesar la imagen y predecir
            processed_image = process_image(pil_image)
            prediction = model.predict(processed_image)[0]
            
            # Obtener la confianza más alta
            max_confidence = np.max(prediction)
            
            # --- FILTRO DE SEGURIDAD (UMBRAL) ---
            # Si la confianza es menor al 40% (0.40), probablemente no sea un perro
            UMBRAL_CONFIANZA = 0.40 

            if max_confidence < UMBRAL_CONFIANZA:
                st.error("⚠️ No hemos podido identificar un perro en esta imagen.")
                st.write(f"El sistema tiene una confianza muy baja ({max_confidence*100:.2f}%).")
                st.info("Por favor, asegúrate de subir una foto clara de un perro.")
                
                # Mostramos la imagen subida de todos modos para que el usuario vea qué falló
                st.image(pil_image, caption="Imagen analizada", width=300)
                
            else:
                # --- SI ES UN PERRO (Confianza > 40%) ---
                # Aquí va todo el código que ya tenías para mostrar resultados
                
                top5_indices = np.argsort(prediction)[-5:][::-1]
                top5_confidences = prediction[top5_indices]
                top5_breed_ids = [class_names[i] for i in top5_indices]
                
                # Predicción principal
                main_breed_id = top5_breed_ids[0]
                main_breed_info = recommendations[main_breed_id]
                main_common_name = main_breed_info["nombre_comun"]

                # --- 5. Mostrar Resultados ---
                st.subheader(f"¡Análisis Completo! 🎯")
                st.header(f"Raza Principal: {main_common_name}")
                
                # (El resto del código de columnas, gráficos y audio sigue igual aquí abajo...)
                # ...
                
                # Para facilitarte la vida, te pego el bloque COMPLETO del 'else' aquí abajo
                # para que puedas copiar y reemplazar todo desde 'if max_confidence < UMBRAL...'
                
                st.write(f"*(Confianza de la predicción: {max_confidence * 100:.2f}%)*")
                
                st.divider()
                
                col1, col2 = st.columns(2)

                with col1:
                    st.image(pil_image, caption="Tu Foto", use_column_width=True)
                    
                    ref_image_path_jpg = f"breed_images/{main_breed_id}.jpg"
                    ref_image_path_png = f"breed_images/{main_breed_id}.png"
                    
                    if os.path.exists(ref_image_path_jpg):
                        st.image(ref_image_path_jpg, caption=f"Referencia: {main_common_name}", use_column_width=True)
                    elif os.path.exists(ref_image_path_png):
                        st.image(ref_image_path_png, caption=f"Referencia: {main_common_name}", use_column_width=True)
                    else:
                        st.warning(f"No se encontró foto de referencia para {main_common_name}.")

                with col2:
                    st.subheader("📊 Gráfico de Similitud (Top 5)")
                    
                    top5_data = {
                        "Raza": [recommendations[bid]["nombre_comun"] for bid in top5_breed_ids],
                        "Confianza": [conf * 100 for conf in top5_confidences]
                    }
                    chart_data = pd.DataFrame(top5_data)
                    
                    st.bar_chart(chart_data, x="Raza", y="Confianza")
                    
                    with st.expander("📋 Ver Guía de Cuidados y Audio", expanded=True):
                        st.write(f"**Temperamento:** {main_breed_info['temperamento']}")
                        st.write(f"**Ejercicio:** {main_breed_info['ejercicio']}")
                        st.write(f"**Cuidado del Pelaje:** {main_breed_info['cuidado']}")
                        
                        st.divider()
                        st.subheader("🔊 Audio-Recomendación")
                        
                        with st.spinner("Generando audio..."):
                            audio_text = f"Recomendaciones para un {main_common_name}. Temperamento: {main_breed_info['temperamento']}. Ejercicio: {main_breed_info['ejercicio']}."
                            audio_file = generate_audio(audio_text)
                            if audio_file:
                                st.audio(audio_file, format='audio/mp3')
        pil_image = Image.open(img_file).convert('RGB')
        
        # --- Animación de Carga (SPINNER) ---
        with st.spinner("Analizando... 🧠"):
            
            # Procesar la imagen y predecir
            processed_image = process_image(pil_image)
            prediction = model.predict(processed_image)[0]
            
            # --- TOP 5 PREDICCIONES ---
            top5_indices = np.argsort(prediction)[-5:][::-1]
            top5_confidences = prediction[top5_indices]
            top5_breed_ids = [class_names[i] for i in top5_indices]
            
            # Predicción principal
            main_breed_id = top5_breed_ids[0]
            main_confidence = top5_confidences[0]
            main_breed_info = recommendations[main_breed_id]
            main_common_name = main_breed_info["nombre_comun"]

        # --- 5. Mostrar Resultados (Layout de 2 Columnas) ---
        st.subheader(f"¡Análisis Completo! 🎯")
        st.header(f"Raza Principal: {main_common_name}")
        st.write(f"*(Confianza de la predicción: {main_confidence * 100:.2f}%)*")
        
        st.divider()
        
        col1, col2 = st.columns(2)

        with col1:
            st.image(pil_image, caption="Tu Foto", use_column_width=True)
            
            # --- Lógica para mostrar la FOTO DE REFERENCIA ---
            ref_image_path_jpg = f"breed_images/{main_breed_id}.jpg"
            ref_image_path_png = f"breed_images/{main_breed_id}.png"
            
            if os.path.exists(ref_image_path_jpg):
                st.image(ref_image_path_jpg, caption=f"Referencia: {main_common_name}", use_column_width=True)
            elif os.path.exists(ref_image_path_png):
                st.image(ref_image_path_png, caption=f"Referencia: {main_common_name}", use_column_width=True)
            else:
                # Si no completaste el Paso 1, mostrará este aviso
                st.warning(f"No se encontró foto de referencia para {main_common_name}.")

        with col2:
            # --- GRÁFICO DE BARRAS TOP 5 ---
            st.subheader("📊 Gráfico de Similitud (Top 5)")
            
            top5_data = {
                "Raza": [recommendations[bid]["nombre_comun"] for bid in top5_breed_ids],
                "Confianza": [conf * 100 for conf in top5_confidences]
            }
            chart_data = pd.DataFrame(top5_data)
            
            st.bar_chart(chart_data, x="Raza", y="Confianza")
            
            # --- Información de Cuidado (EXPANDER) ---
            with st.expander("📋 Ver Guía de Cuidados y Audio", expanded=True):
                st.write(f"**Temperamento:** {main_breed_info['temperamento']}")
                st.write(f"**Ejercicio:** {main_breed_info['ejercicio']}")
                st.write(f"**Cuidado del Pelaje:** {main_breed_info['cuidado']}")
                
                st.divider()
                st.subheader("🔊 Audio-Recomendación")
                
                with st.spinner("Generando audio..."):
                    audio_text = f"Recomendaciones para un {main_common_name}. Temperamento: {main_breed_info['temperamento']}. Ejercicio: {main_breed_info['ejercicio']}."
                    audio_file = generate_audio(audio_text)
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')

# --- Ejecutar la app ---
if __name__ == "__main__":
    if model and recommendations:
        main()
    else:
        st.error("La aplicación no se puede iniciar. Revisa los archivos del modelo y JSON.")