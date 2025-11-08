import re
import nltk
from nltk.corpus import stopwords
import warnings
import spacy

# --- Configuración Inicial (Descarga de recursos) ---
try:
    # Descargar la lista de stopwords en español
    nltk.data.find('corpora/stopwords')
    print("Recurso 'stopwords' de NLTK ya está descargado.")
except LookupError:
    print("Descargando recurso 'stopwords' de NLTK...")
    nltk.download('stopwords')

# --- Definición de Recursos ---

# 1. Stopwords en Español
try:
    STOPWORDS_ES = set(stopwords.words('spanish'))
    print(f"Cargadas {len(STOPWORDS_ES)} stopwords en español.")
except Exception as e:
    warnings.warn(f"No se pudieron cargar las stopwords en español. Error: {e}")
    STOPWORDS_ES = set()

# 2. Modelo de spaCy
try:
    NLP_SPACY = spacy.load('es_core_news_sm')
    print("Modelo 'es_core_news_sm' de spaCy cargado.")
except IOError:
    warnings.warn("ERROR: Modelo 'es_core_news_sm' de spaCy no encontrado.")
    warnings.warn("Ejecuta: python -m spacy download es_core_news_sm")
    NLP_SPACY = None

# --- Funciones de Normalización ---

def pipeline_a_normalize(text: str) -> str:
    """
    Pipeline A: Tokenización y Limpieza Básica.
    1. Convierte a minúsculas.
    2. Elimina caracteres que NO sean letras del alfabeto español (incl. tildes y ñ) o espacios.
    3. Elimina espacios en blanco extra.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convertir a minúsculas
    text = text.lower()
    
    # 2. Eliminar caracteres no alfabéticos (mantiene letras, tildes, ñ y espacios)
    #    [^...] significa "cualquier caracter que NO esté en este conjunto"
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    
    # 3. Eliminar espacios en blanco extra (creados por las eliminaciones)
    #    .split() divide por espacios, y .join() vuelve a unir con un solo espacio
    text = " ".join(text.split())
    
    return text


def pipeline_b_normalize(text: str) -> str:
    """
    Pipeline B: Pipeline A + Eliminación de Stopwords.
    1. Aplica toda la limpieza del Pipeline A.
    2. Tokeniza (divide) el texto en palabras.
    3. Elimina las palabras que están en la lista de STOPWORDS_ES.
    4. Vuelve a unir el texto.
    """
    # 1. Reutilizar el Pipeline A
    cleaned_text = pipeline_a_normalize(text)
    
    # 2. Tokenizar
    words = cleaned_text.split()
    
    # 3. Eliminar stopwords
    if not STOPWORDS_ES:
        warnings.warn("Lista de stopwords vacía, no se filtrará nada.")
        return cleaned_text
        
    filtered_words = [word for word in words if word not in STOPWORDS_ES]
    
    # 4. Vuelve a unir el texto
    return " ".join(filtered_words)

# --- Puedes añadir los pipelines C y D aquí abajo ---

def pipeline_c_normalize(text: str) -> str:
    """
    Pipeline C: Pipeline B + Lematización.
    1. Aplica la limpieza y filtrado de stopwords del Pipeline B.
    2. Utiliza Spacy para convertir cada palabra a su forma raíz (lema).
    """
    if NLP_SPACY is None:
        warnings.warn("Spacy no está cargado. Saltando lematización. Devoviendo salida de Pipeline B.")
        return pipeline_b_normalize(text)

    # 1. Aplicar Pipeline B
    text_no_stopwords = pipeline_b_normalize(text)
    
    # 2. Procesar el texto con Spacy
    # Desactivar 'parser' y 'ner' acelera el proceso, solo necesitamos el 'lemmatizer'.
    doc = NLP_SPACY(text_no_stopwords, disable=['parser', 'ner'])
    
    # 3. Extraer el lema (forma raíz) de cada token
    lemmatized_words = [token.lemma_ for token in doc if token.lemma_]
    
    return " ".join(lemmatized_words)


# --- Bloque de Prueba ---
if __name__ == "__main__":
    
    ejemplo = "El hotel no era muy bueno, tampoco me gustó la comida. ¡Nunca volveré! 🤬"
    
    print("--- PRUEBA DE PIPELINES ---")
    print(f"Original:   {ejemplo}\n")
    
    print(f"Pipeline A: {pipeline_a_normalize(ejemplo)}")
    print(f"Pipeline B: {pipeline_b_normalize(ejemplo)}")
    print(f"Pipeline C: {pipeline_c_normalize(ejemplo)}")