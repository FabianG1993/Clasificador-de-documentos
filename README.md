# 📄 Clasificador inteligente de documentos (Document AI) 📄

Un pipeline de Procesamiento Inteligente de Documentos (IDP) de extremo a extremo, diseñado para clasificar automáticamente documentos corporativos (facturas y contratos) extrayendo su texto mediante Visión por Computadora y aplicando Machine Learning.

## Arquitectura

Este proyecto resuelve el cuello de botella operativo de la clasificación manual de documentos. Su principal diferenciador es la **privacidad absoluta (Zero-API)**: todo el procesamiento (OCR y ML) se ejecuta 100% en local, sin enviar datos sensibles a la nube, ideal para sectores regulados (banca, legal, finanzas).

**Stack tecnológico:**
* **Ingesta visual:** `OpenCV` (Preprocesamiento) + `Tesseract OCR` (Extracción de texto).
* **NLP & Representación:** `TF-IDF` (Vectorización semántica basada en la frecuencia de términos del negocio).
* **Machine Learning:** `Random Forest Classifier` (Algoritmo explicable y auditable).
* **Interfaz de usuario:** `Streamlit`.

## Instrucciones

Este proyecto requiere dependencias tanto a nivel de sistema operativo como de Python.

### 1. Instalación del motor OCR (requisito crítico)
El código utiliza Tesseract bajo el capó. Debes instalarlo en tu sistema operativo:
* **Windows:** Descarga el instalador desde [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). 
  * *Importante:* Durante la instalación, asegúrate de marcar la casilla para descargar los datos de idioma **"Spanish"**.
  * Si lo instalas en una ruta distinta a `C:\Program Files\Tesseract-OCR\tesseract.exe`, actualiza la línea 9 en `ocr_engine.py`.
* **Linux/Mac:** Usar `sudo apt-get install tesseract-ocr tesseract-ocr-spa` o `brew install tesseract`.

### 2. Entorno Python
Clona este repositorio e instala las librerías:
```bash
pip install -r requirements.txt
pip install scikit-learn joblib streamlit
```

### 3.Uso del aplicativo
Para entrenar el modelo desde cero con tus propios documentos, coloca imágenes en la carpeta dataset/ (ej. dataset/facturas/, dataset/contratos/) y ejecuta:
```bash
python classifier.py
```
Para lanzar la interfaz visual interactiva y probar documentos nuevos, ejecuta:
```bash
python -m streamlit run app_clasificador.py
```

### 4.NOTA: Entorno de producción
El modelo .pkl incluido es una Prueba de Concepto (PoC) arquitectónica entrenada con una muestra reducida de documentos para demostrar el flujo E2E. Para un despliegue en producción real, se recomienda:

- Ingestar un corpus balanceado de +500 documentos por categoría.
- Implementar técnicas de Data Augmentation (rotación, ruido, blur) durante el entrenamiento para robustecer el modelo frente a imagenes/escaneos de baja calidad.
