"""
classifier.py - Clasificador de documentos (Facturas vs Contratos)

Pipeline:
1. Recorre las carpetas del dataset y extrae texto con OCR.
2. Entrena un modelo TF-IDF + SVM con los textos etiquetados.
3. Guarda el modelo entrenado como .pkl en la carpeta del proyecto.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

from ocr_engine import MotorOCR


# --- CONFIGURACION -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELO_PATH = os.path.join(BASE_DIR, "clasificador_documentos.pkl")

# Mapeo: nombre de carpeta -> etiqueta
CATEGORIAS = {
    "Ej. Facturas": "factura",
    "Ej.Contratos": "contrato",
}

EXTENSIONES_VALIDAS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


# --- FUNCIONES ---------------------------------------------------------------

def recolectar_datos(motor_ocr):
    """
    Recorre cada subcarpeta del dataset, extrae el texto de cada imagen
    con OCR y devuelve dos listas paralelas: textos y etiquetas.
    """
    textos = []
    etiquetas = []

    for carpeta, etiqueta in CATEGORIAS.items():
        ruta_carpeta = os.path.join(DATASET_DIR, carpeta)

        if not os.path.isdir(ruta_carpeta):
            print(f"[WARN] Carpeta no encontrada, saltando: {ruta_carpeta}")
            continue

        archivos = [
            f for f in os.listdir(ruta_carpeta)
            if f.lower().endswith(EXTENSIONES_VALIDAS)
        ]

        if not archivos:
            print(f"[WARN] Sin imagenes validas en: {ruta_carpeta}")
            continue

        print(f"\n--- Procesando '{carpeta}' ({len(archivos)} imagenes) -> etiqueta: '{etiqueta}'")

        for nombre_archivo in archivos:
            ruta_imagen = os.path.join(ruta_carpeta, nombre_archivo)
            print(f"   OCR en: {nombre_archivo} ... ", end="")

            texto = motor_ocr.extraer_texto(ruta_imagen)

            if texto and not texto.startswith("Error procesando"):
                textos.append(texto)
                etiquetas.append(etiqueta)
                print(f"OK ({len(texto)} caracteres)")
            else:
                print(f"FALLO -> {texto[:80] if texto else 'sin texto'}")

    return textos, etiquetas


def entrenar_modelo(textos, etiquetas):
    """
    Crea un pipeline TF-IDF + LinearSVC y lo entrena con los textos.
    Retorna el pipeline entrenado.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", LinearSVC(
            C=1.0,
            max_iter=10000,
            class_weight="balanced",
        )),
    ])

    print("\n[INFO] Entrenando modelo...")
    pipeline.fit(textos, etiquetas)

    # Validacion cruzada si hay suficientes datos
    n_muestras = len(textos)
    if n_muestras >= 4:
        n_folds = min(5, n_muestras)
        scores = cross_val_score(pipeline, textos, etiquetas, cv=n_folds, scoring="accuracy")
        print(f"   Validacion cruzada ({n_folds}-fold): {scores.mean():.2%} +/- {scores.std():.2%}")
    else:
        print("   Pocos datos para validacion cruzada, se entreno con todos.")

    # Re-entrenar con TODOS los datos para el modelo final
    pipeline.fit(textos, etiquetas)
    return pipeline


def guardar_modelo(pipeline):
    """Guarda el pipeline entrenado como archivo .pkl."""
    joblib.dump(pipeline, MODELO_PATH)
    tamano_mb = os.path.getsize(MODELO_PATH) / (1024 * 1024)
    print(f"\n[OK] Modelo guardado en: {MODELO_PATH}")
    print(f"     Tamano: {tamano_mb:.2f} MB")


def predecir(texto, pipeline=None):
    """
    Clasifica un texto dado. Si no se pasa pipeline, carga el .pkl guardado.
    """
    if pipeline is None:
        if not os.path.exists(MODELO_PATH):
            raise FileNotFoundError(
                f"No se encontro el modelo en '{MODELO_PATH}'. "
                "Ejecuta primero el entrenamiento."
            )
        pipeline = joblib.load(MODELO_PATH)

    prediccion = pipeline.predict([texto])[0]
    return prediccion


# --- EJECUCION PRINCIPAL -----------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  CLASIFICADOR DE DOCUMENTOS - Entrenamiento")
    print("=" * 60)

    # 1. Inicializar motor OCR
    motor = MotorOCR(idioma="spa")

    # 2. Recolectar datos del dataset
    textos, etiquetas = recolectar_datos(motor)

    if len(textos) < 2:
        print("\n[ERROR] Se necesitan al menos 2 documentos con texto para entrenar.")
        print("        Revisa que las imagenes del dataset sean legibles.")
        exit(1)

    clases_unicas = set(etiquetas)
    if len(clases_unicas) < 2:
        print(f"\n[ERROR] Solo se encontro la clase '{clases_unicas.pop()}'.")
        print("        Se necesitan al menos 2 clases distintas para entrenar.")
        exit(1)

    print(f"\n[OK] Datos recolectados: {len(textos)} documentos")
    for clase in sorted(clases_unicas):
        count = etiquetas.count(clase)
        print(f"     - {clase}: {count} documentos")

    # 3. Entrenar
    modelo = entrenar_modelo(textos, etiquetas)

    # 4. Guardar el .pkl
    guardar_modelo(modelo)

    # 5. Prueba rapida con los primeros textos del dataset
    print("\n" + "=" * 60)
    print("  PRUEBA RAPIDA")
    print("=" * 60)
    for i, (txt, etq) in enumerate(zip(textos[:2], etiquetas[:2])):
        pred = predecir(txt, modelo)
        estado = "OK" if pred == etq else "FALLO"
        print(f"   [{estado}] Doc {i+1}: real='{etq}' -> predicho='{pred}'")

    print("\n[LISTO] Entrenamiento completado exitosamente!")
    print(f"        Usa 'predecir(texto)' para clasificar nuevos documentos.")