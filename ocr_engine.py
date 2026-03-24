import cv2
import pytesseract
from PIL import Image
import os

# --- CONFIGURACIÓN CRÍTICA PARA WINDOWS ---
# Reemplaza esta ruta si instalaste Tesseract en otro lado.
# La 'r' al inicio es crucial para que Windows lea bien las barras invertidas.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class MotorOCR:
    """Clase encargada de preprocesar imágenes y extraer texto usando Tesseract."""
    
    def __init__(self, idioma='spa'):
        # Usamos 'spa' para español. Si tus facturas están en inglés, usa 'eng'.
        self.idioma = idioma
        print(f"Motor OCR inicializado en idioma: '{self.idioma}'")

    def preprocesar_imagen(self, ruta_imagen):
        """Mejora la imagen antes de leerla para aumentar la precisión del OCR."""
        if not os.path.exists(ruta_imagen):
            raise FileNotFoundError(f"No se encontró la imagen: {ruta_imagen}")

        # 1. Leer imagen con OpenCV
        img = cv2.imread(ruta_imagen)
        
        # 2. Convertir a escala de grises (El color confunde al OCR)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Aplicar umbral adaptativo (Binarización: Fondo blanco puro, letras negras puras)
        # Esto es vital para facturas arrugadas, escaneos oscuros o fotos con sombras.
        img_binarizada = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
        return img_binarizada

    def extraer_texto(self, ruta_imagen):
        """Ejecuta el pipeline completo y devuelve el texto crudo."""
        try:
            # Preprocesamos para limpiar el ruido
            img_limpia = self.preprocesar_imagen(ruta_imagen)
            
            # Pasamos la imagen limpia a Tesseract
            texto = pytesseract.image_to_string(img_limpia, lang=self.idioma)
            return texto.strip()
            
        except Exception as e:
            return f"Error procesando el documento: {e}"

# --- PRUEBA LOCAL ---
if __name__ == "__main__":
    motor = MotorOCR(idioma='spa')
    
    # ⚠️ INSTRUCCIÓN: Cambia esto por la ruta de una foto de una factura real que tengas
    imagen_prueba = "Factura1.png" 
    
    # Crear un archivo de texto para ver la salida cómodamente
    if os.path.exists(imagen_prueba):
        print("Procesando documento...")
        texto_extraido = motor.extraer_texto(imagen_prueba)
        
        print("\n--- TEXTO EXTRAÍDO ---")
        print(texto_extraido)
        print("----------------------")
        
        # Guardamos el resultado en un txt para analizarlo mejor
        with open("resultado_ocr.txt", "w", encoding="utf-8") as f:
            f.write(texto_extraido)
            print("\nResultado guardado en 'resultado_ocr.txt'")
    else:
        print(f"Por favor, coloca una imagen llamada '{imagen_prueba}' en esta carpeta para probar.")