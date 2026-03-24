import streamlit as st
import joblib
import os
from PIL import Image
from ocr_engine import MotorOCR

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Clasificador documental B2B", layout="wide")
MODEL_PATH = "clasificador_documentos.pkl"
TEMP_IMG_PATH = "temp_upload.jpg"

# --- INICIALIZACIÓN (Caché para velocidad) ---
@st.cache_resource
def cargar_modelo_y_ocr():
    """Carga el motor OCR y el modelo ML una sola vez en memoria."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    motor_ocr = MotorOCR(idioma='spa')
    modelo_ml = joblib.load(MODEL_PATH)
    return motor_ocr, modelo_ml

motor, modelo = cargar_modelo_y_ocr()

# --- INTERFAZ VISUAL ---
st.title("📄 CLASIFICADOR DE DOCUMENTOS 📄")
st.markdown("""
Esta herramienta automatiza la clasificación de documentos financieros (facturas y contratos). 
Sube un escaneo o foto, y el sistema extraerá el texto mediante **OCR** para clasificarlo 
usando **Machine Learning (TF-IDF + Random Forest)**.
""")

if modelo is None:
    st.error(f"No se encontró el modelo entrenado '{MODEL_PATH}'. Ejecuta 'classifier.py' primero.")
    st.stop()

# --- ZONA DE CARGA ---
st.header("1. Cargar documento")
archivo_subido = st.file_uploader("Arrastra una imagen (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

if archivo_subido is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Documento original")
        # Mostrar la imagen
        img = Image.open(archivo_subido)
        st.image(img, use_container_width=True)
        
        # Guardar temporalmente para que OpenCV/Tesseract la puedan leer
        with open(TEMP_IMG_PATH, "wb") as f:
            f.write(archivo_subido.getbuffer())

    with col2:
        st.subheader("Análisis")
        
        with st.spinner("1️⃣ Extrayendo texto con OCR (Tesseract)..."):
            texto_extraido = motor.extraer_texto(TEMP_IMG_PATH)
            
        if not texto_extraido or len(texto_extraido.strip()) < 10:
            st.warning("No se pudo extraer suficiente texto legible de esta imagen.")
        else:
            with st.spinner("2️⃣ Clasificando semánticamente (Machine Learning)..."):
                # Predecir la categoría (Esto siempre funciona)
                prediccion = modelo.predict([texto_extraido])[0]
                
                # --- NUEVA LÓGICA DEFENSIVA ---
                st.success(f"**Clasificación:** {prediccion.upper()}")
                
                # Verificamos si el modelo matemático soporta probabilidades
                if hasattr(modelo, "predict_proba"):
                    try:
                        probabilidades = modelo.predict_proba([texto_extraido])[0]
                        confianza = max(probabilidades) * 100
                        st.info(f"**Nivel de Confianza:** {confianza:.2f}%")
                    except Exception:
                        pass # Si falla internamente, simplemente no mostramos el cuadro azul
                # -----------------------------
            
            # Acordeón para mostrar el texto extraído (Transparencia para el cliente)
            with st.expander("Ver texto extraído por el OCR (Auditoría)"):
                st.text(texto_extraido)

# Limpieza del archivo temporal al salir
if os.path.exists(TEMP_IMG_PATH) and archivo_subido is None:
    try:
        os.remove(TEMP_IMG_PATH)
    except:
        pass