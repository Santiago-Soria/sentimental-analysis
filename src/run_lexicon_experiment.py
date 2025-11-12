import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import os
import warnings

# --- Importar la normalización ganadora (Pipeline D) ---
try:
    from normalization_functions import pipeline_d_normalize
except ImportError:
    import sys
    sys.path.append('.')
    from src.normalization_functions import pipeline_d_normalize

# --- Rutas de Archivos ---
# Ajusta estas rutas según donde tengas tus archivos descargados
PATH_SEL = "../data/lexicons/SEL_full.txt"
PATH_LIWC = "../data/lexicons/LIWC2007.dic"
PATH_EMOJIS = "../data/lexicons/Emojis lexicon.XLSX" 
TRAIN_PATH = "../data/processed/train.csv"


class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Diccionarios para mapeo rápido
        self.sel_dict = {}   # word -> 'pos' | 'neg'
        self.liwc_pos = set()
        self.liwc_neg = set()
        self.emoji_scores = {} # emoji -> {'pos': val, 'neg': val}
        
    def load_resources(self):
        print("--- Cargando recursos lingüísticos ---")
        
        # 1. Cargar SEL (Spanish Emotion Lexicon)
        # Estructura: Palabra, Nula[%], Baja[%], Media[%], Alta[%], PFA, Categoría
        try:
            # Asumimos separación por tabuladores ('\t') dado el formato .txt usual de SEL
            # Si falla, intentar sep='\s+' para espacios múltiples
            sel_df = pd.read_csv(PATH_SEL, sep='\t', encoding='latin-1', skiprows=1, header=None)
            
            # Mapeo de categorías SEL a Polaridad Binaria
            # Alegría, Sorpresa -> Positivo
            # Enojo, Miedo, Repulsión, Tristeza -> Negativo
            pos_cats = ['Alegría', 'Sorpresa']
            neg_cats = ['Enojo', 'Miedo', 'Repulsión', 'Tristeza']

            for _, row in sel_df.iterrows():
                # Columna 0: Palabra, Columna 6: Categoría
                word = str(row[0]).lower().strip()
                cat = str(row[6]).strip()
                
                if cat in pos_cats:
                    self.sel_dict[word] = 'pos'
                elif cat in neg_cats:
                    self.sel_dict[word] = 'neg'
            
            print(f"-> SEL cargado: {len(self.sel_dict)} palabras.")
        except Exception as e:
            print(f"Warning: Error cargando SEL: {e}")

        # 2. Cargar LIWC
        try:
            with open(PATH_LIWC, 'r', encoding='latin-1') as f:
                is_body = False
                count = 0
                for line in f:
                    line = line.strip()
                    if line == '%':
                        is_body = not is_body
                        continue
                    if is_body:
                        parts = line.split()
                        if len(parts) > 1:
                            word = parts[0].replace('*', '') # Quitar comodines simples
                            cats = parts[1:]
                            # IDs confirmados por ti: 126 (Pos), 127 (Neg)
                            if '126' in cats: self.liwc_pos.add(word)
                            if '127' in cats: self.liwc_neg.add(word)
                            count += 1
            print(f"-> LIWC cargado: {count} entradas procesadas.")
        except Exception as e:
            print(f"Warning: Error cargando LIWC: {e}")

        # 3. Cargar Emojis Lexicon (XLSX)
        # Columnas: id, name, emoji, ..., negative, positive
        try:
            # Requiere 'openpyxl' instalado
            emoji_df = pd.read_excel(PATH_EMOJIS)
            
            for _, row in emoji_df.iterrows():
                emoji_char = str(row['emoji']).strip()
                
                # Obtener valores de las columnas 'positive' y 'negative'
                # Asumimos que son numéricos (ej. 1 o 0, o una probabilidad)
                pos_val = row.get('positive', 0)
                neg_val = row.get('negative', 0)
                
                self.emoji_scores[emoji_char] = {
                    'pos': float(pos_val) if pd.notnull(pos_val) else 0.0,
                    'neg': float(neg_val) if pd.notnull(neg_val) else 0.0
                }
            print(f"-> Emojis cargados: {len(self.emoji_scores)} emojis.")
            
        except Exception as e:
            print(f"Warning: Error cargando Emojis: {e}")

    def fit(self, X, y=None):
        self.load_resources()
        return self

    def transform(self, X):
        # X es una lista/serie de textos NORMALIZADOS (con Pipeline D)
        # Importante: Pipeline D conserva negaciones, pero los léxicos no suelen tenerlas.
        # Haremos coincidencia exacta de palabras.
        
        features = []
        
        for text in X:
            # Inicializar contadores para este documento
            feats = {
                'sel_pos': 0, 'sel_neg': 0,
                'liwc_pos': 0, 'liwc_neg': 0,
                'emoji_pos': 0, 'emoji_neg': 0
            }
            
            # Tokenizar por espacio (asumiendo que ya pasó por normalización básica)
            words = text.split()
            
            for word in words:
                # 1. Checar SEL
                # Si el pipeline D generó "NO_bueno", 'word' es "NO_bueno".
                # El diccionario tiene "bueno". Necesitamos limpiar el prefijo para buscar en el diccionario.
                clean_word = word.replace("NO_", "")
                is_negated = word.startswith("NO_")

                # --- Lógica SEL ---
                if clean_word in self.sel_dict:
                    polarity = self.sel_dict[clean_word]
                    if is_negated:
                        # Invertir polaridad si está negada
                        if polarity == 'pos': feats['sel_neg'] += 1
                        else: feats['sel_pos'] += 1
                    else:
                        if polarity == 'pos': feats['sel_pos'] += 1
                        else: feats['sel_neg'] += 1
                
                # --- Lógica LIWC ---
                if clean_word in self.liwc_pos:
                    if is_negated: feats['liwc_neg'] += 1
                    else: feats['liwc_pos'] += 1
                elif clean_word in self.liwc_neg:
                    if is_negated: feats['liwc_pos'] += 1
                    else: feats['liwc_neg'] += 1
                
                # --- Lógica Emojis ---
                # Los emojis no suelen ser negados por "NO_", pero checamos el token directo
                if word in self.emoji_scores:
                    vals = self.emoji_scores[word]
                    feats['emoji_pos'] += vals['pos']
                    feats['emoji_neg'] += vals['neg']
            
            # Normalizar por longitud del documento (evita sesgo por textos largos)
            doc_len = len(words) if len(words) > 0 else 1
            
            features.append([
                feats['sel_pos'] / doc_len,
                feats['sel_neg'] / doc_len,
                feats['liwc_pos'] / doc_len,
                feats['liwc_neg'] / doc_len,
                feats['emoji_pos'], # Emojis se suelen dejar como conteo o suma bruta
                feats['emoji_neg']
            ])
            
        return np.array(features)


# --- EJECUCIÓN PRINCIPAL ---

if __name__ == "__main__":
    print("Cargando datos de entrenamiento...")
    df_train = pd.read_csv(TRAIN_PATH)
    
    # 1. Normalizar con Pipeline D (El ganador)
    print("Aplicando Pipeline D...")
    X_train_norm = df_train['text'].apply(pipeline_d_normalize)
    y_train = df_train['Polarity']
    
    # 2. Crear Feature Union
    # Rama A: Bolsa de Palabras (Binary) - Lo que ya funcionaba
    # Rama B: Características de Léxico - Lo nuevo
    combined_features = FeatureUnion([
        ('bow', CountVectorizer(binary=False)),
        ('lexicons', LexiconFeatureExtractor())
    ])
    
    # 3. Crear Pipeline Final con Modelo
    # Usamos LogisticRegression porque fue el mejor modelo simple
    model_pipeline = Pipeline([
        ('features', combined_features),
        ('classifier', LogisticRegression(max_iter=2000))
    ])
    
    # 4. Evaluar
    print("\nEvaluando modelo con características aumentadas (Feature Union)...")
    cv_scores = cross_validate(
        model_pipeline, 
        X_train_norm, 
        y_train, 
        cv=5, 
        scoring='f1_macro',
        n_jobs=-1
    )
    
    mean_score = cv_scores['test_score'].mean()
    print(f"\n>>> F1-Macro Promedio con Léxicos: {mean_score:.4f} <<<")
    print("(Compara este valor con tu mejor resultado anterior de ~0.455)")