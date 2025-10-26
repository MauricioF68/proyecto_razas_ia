import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
from gtts import gTTS
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Dog-ID: Identificador de Razas",
    page_icon="🐶",
    layout="wide"
)

# --- 1. Cargar el Modelo y los Datos ---

# Usamos st.cache_resource para cargar el modelo solo una vez
@st.cache_resource
def load_my_model():
    """Carga el modelo Keras entrenado."""
    try:
        model = tf.keras.models.load_model('dog_breed_classifier_v2_120_breeds.keras')
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Usamos st.cache_data para cargar las recomendaciones solo una vez
@st.cache_data
def load_recommendations():
    """Carga el archivo JSON de recomendaciones."""
    try:
        with open('recommendations.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error al cargar el archivo JSON: {e}")
        return None

model = load_my_model()
recommendations = load_recommendations()

# Obtenemos la lista de nombres de clases (los IDs) desde las claves del JSON
# Esto debe coincidir con el orden en que se entrenó el modelo (alfabético)
if recommendations:
    class_names = list(recommendations.keys())
else:
    class_names = []

# --- 2. Funciones Auxiliares ---

def process_image(pil_image):
    """Convierte una imagen PIL al formato que el modelo necesita."""
    # Convertir a array de numpy
    img = np.array(pil_image)
    
    # Redimensionar la imagen a 224x224
    img_resized = tf.image.resize(img, (224, 224))
    
    # Añadir una dimensión de "batch" (lote)
    img_expanded = tf.expand_dims(img_resized, axis=0)
    
    return img_expanded

def generate_audio(text):
    """Genera un archivo de audio en memoria a partir del texto."""
    try:
        # Generar el audio en español
        tts = gTTS(text=text, lang='es', slow=False)
        
        # Crear un objeto de bytes en memoria para guardar el audio
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        
        # Regresar al inicio del archivo en memoria para que st.audio pueda leerlo
        audio_fp.seek(0)
        return audio_fp
    except Exception as e:
        st.error(f"Error al generar el audio: {e}")
        return None

# --- 3. Interfaz de Usuario de Streamlit ---

st.title("🐶 Dog-ID: Identificador de Razas")
st.write("¡Usa la cámara de tu dispositivo para identificar la raza de tu perro y obtener consejos de cuidado!")

# Verificamos si el modelo y los datos se cargaron correctamente
if model is None or recommendations is None:
    st.error("La aplicación no se pudo iniciar. Faltan archivos esenciales.")
else:
    # El componente de cámara de Streamlit
    img_file_buffer = st.camera_input("Toma una foto para analizarla:")

    if img_file_buffer is not None:
        # El usuario ha tomado una foto
        
        # 1. Abrir la imagen capturada
        pil_image = Image.open(img_file_buffer)
        
        # 2. Mostrar la imagen al usuario
        st.image(pil_image, caption="Foto capturada. Analizando...", use_column_width=True)

        # 3. Procesar la imagen y predecir (con un indicador de "cargando")
        with st.spinner("Analizando la raza... 🧠"):
            
            # Procesar la imagen
            processed_image = process_image(pil_image)
            
            # Hacer la predicción
            prediction = model.predict(processed_image)
            
            # Obtener la predicción principal
            pred_index = np.argmax(prediction)
            pred_confidence = np.max(prediction)
            
            # Mapear el índice al ID de la raza
            breed_id = class_names[pred_index]
            
            # Obtener la información de nuestro JSON
            breed_info = recommendations[breed_id]
            common_name = breed_info["nombre_comun"]
            temperament = breed_info["temperamento"]
            exercise = breed_info["ejercicio"]
            care = breed_info["cuidado"]

        # 4. Mostrar los resultados
        st.subheader(f"¡Creo que es un... {common_name}!")
        st.write(f"**Confianza de la predicción:** {pred_confidence * 100:.2f}%")
        
        # Usamos columnas para un mejor diseño
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Guía de Cuidados 📋")
            st.write(f"**Temperamento:** {temperament}")
            st.write(f"**Ejercicio:** {exercise}")
            st.write(f"**Cuidado del Pelaje:** {care}")

        # 5. Generar y mostrar el audio
        with col2:
            st.subheader("Escuchar Recomendación 🔊")
            with st.spinner("Generando audio..."):
                # Crear el texto para el audio
                audio_text = f"Recomendaciones para un {common_name}. Temperamento: {temperament}. Ejercicio: {exercise}. Cuidado del pelaje: {care}."
                
                # Generar el archivo de audio en memoria
                audio_file = generate_audio(audio_text)
                
                if audio_file:
                    # Mostrar el reproductor de audio
                    st.audio(audio_file, format='audio/mp3')