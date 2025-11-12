import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from normalization_functions import pipeline_d_normalize

# Recurso nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpus/stopwords')
except LookupError:
    nltk.download('stopwords')

# Carga las stopwords en español
stop_words_es = set(stopwords.words('spanish'))

# Config rutas
SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

# Apunta al archivo de entrenamiento creado
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")

# Pipeline D: Tokenización + Limpieza + Stopwords + Negaciones
def preprocesador_pipeline_d(texto):
    if not isinstance(texto, str):
        return ""
        
    texto = texto.lower()
    
    # Manejo de negaciones
    # Unimos "no", "ni", "nunca" con la palabra que le sigue
    texto = re.sub(r'\b(no|ni|nunca)\s+(\w+)', r'\1_\2', texto)
    
    # Tokenización y eliminación de no-alfanuméricos
    # Encuentra solo palabras
    tokens = re.findall(r'\b\w+\b', texto)
    
    # Eliminación de stopwords
    tokens_limpios = [palabra for palabra in tokens if palabra not in stop_words_es]
    
    # Se retorna el texto reconstruido para el Vectorizer
    return " ".join(tokens_limpios)

# Main function
def main():
    print(f"Cargando datos de {TRAIN_PATH}...")
    try:
        train_df = pd.read_csv(TRAIN_PATH)
    except FileNotFoundError:
        print(f"Error: No se encontró {TRAIN_PATH}")
        print("Asegúrate de ejecutar '0_split_data.py' primero.")
        return

    # Definir X_train y y_train
    X_train = train_df['text'].fillna('') 
    y_train = train_df['Polarity']
    
    print(f"Datos de entrenamiento cargados: {len(X_train)} filas.")
    print("Pipeline D")

    # Definimos las 3 representaciones
    representaciones = {
        "Binaria": CountVectorizer(preprocessor=pipeline_d_normalize, binary=True),
        "Frecuencia": CountVectorizer(preprocessor=pipeline_d_normalize, binary=False),
        "TF-IDF": TfidfVectorizer(preprocessor=pipeline_d_normalize)
    }

    # Definimos los 2 modelos
    modelos = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=0)
    }

    # Bucle de experimentación
    resultados_f1_macro = {}

    for nombre_rep, vectorizador in representaciones.items():
        for nombre_mod, modelo in modelos.items():
            
            # Creamos el pipeline final
            pipeline_completo = Pipeline([
                ('vectorizador', vectorizador),
                ('clasificador', modelo)
            ])
            
            # Ejecutamos la validación cruzada (cv=5)
            nombre_experimento = f"D: {nombre_rep} + {nombre_mod}"
            print(f"--- Probando: {nombre_experimento} ---")
            
            cv_resultados = cross_validate(
                pipeline_completo,
                X_train, 
                y_train, 
                cv=5, 
                scoring='f1_macro', # lo importante del pipeline
                n_jobs=-1 
            )
            
            # Guardamos el F1-Macro promedio
            f1_promedio = np.mean(cv_resultados['test_score'])
            resultados_f1_macro[nombre_experimento] = f1_promedio
            print(f"Resultado F1-Macro: {f1_promedio:.4f}\n")
    
    # Resultados finales
    print("Resultados Pipeline D:")
    
    df_resultados = pd.DataFrame.from_dict(
        resultados_f1_macro, 
        orient='index', 
        columns=['F1-Macro Promedio']
    )
    print(df_resultados.sort_values(by='F1-Macro Promedio', ascending=False))


if __name__ == "__main__":
    main()