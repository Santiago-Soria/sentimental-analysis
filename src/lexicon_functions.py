import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
import warnings

# --- 1. Gestión Dinámica de Rutas ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "lexicons")

PATH_SEL = os.path.join(DATA_DIR, "SEL_full.txt")
PATH_LIWC = os.path.join(DATA_DIR, "LIWC2007.dic")
PATH_EMOJIS = os.path.join(DATA_DIR, "Emojis lexicon.XLSX") # Verifica mayúsculas/minúsculas

class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sel_dict = {}
        self.liwc_pos = set()
        self.liwc_neg = set()
        self.emoji_scores = {}
        self._resources_loaded = False # Evitar recargas innecesarias

    def load_resources(self):
        if self._resources_loaded:
            return

        print("--- [lexicon_utils] Cargando recursos lingüísticos ---")
        
        # 1. Cargar SEL (TXT)
        try:
            sel_df = pd.read_csv(PATH_SEL, sep='\t', encoding='latin-1', skiprows=1, header=None)
            pos_cats = ['Alegría', 'Sorpresa']
            neg_cats = ['Enojo', 'Miedo', 'Repulsión', 'Tristeza']

            for _, row in sel_df.iterrows():
                word = str(row[0]).lower().strip()
                cat = str(row[6]).strip()
                if cat in pos_cats: self.sel_dict[word] = 'pos'
                elif cat in neg_cats: self.sel_dict[word] = 'neg'
        except Exception as e:
            warnings.warn(f"Error cargando SEL en {PATH_SEL}: {e}")

        # 2. Cargar LIWC (DIC)
        try:
            with open(PATH_LIWC, 'r', encoding='latin-1') as f:
                is_body = False
                for line in f:
                    line = line.strip()
                    if line == '%':
                        is_body = not is_body
                        continue
                    if is_body:
                        parts = line.split()
                        if len(parts) > 1:
                            word = parts[0].replace('*', '')
                            cats = parts[1:]
                            if '126' in cats: self.liwc_pos.add(word)
                            if '127' in cats: self.liwc_neg.add(word)
        except Exception as e:
            warnings.warn(f"Error cargando LIWC en {PATH_LIWC}: {e}")

        # 3. Cargar Emojis (XLSX)
        try:
            emoji_df = pd.read_excel(PATH_EMOJIS)
            for _, row in emoji_df.iterrows():
                emoji_char = str(row['emoji']).strip()
                pos_val = row.get('positive', 0)
                neg_val = row.get('negative', 0)
                self.emoji_scores[emoji_char] = {
                    'pos': float(pos_val) if pd.notnull(pos_val) else 0.0,
                    'neg': float(neg_val) if pd.notnull(neg_val) else 0.0
                }
        except Exception as e:
            warnings.warn(f"Error cargando Emojis en {PATH_EMOJIS}: {e}")
            
        self._resources_loaded = True

    def fit(self, X, y=None):
        self.load_resources()
        return self

    def transform(self, X):
        features = []
        for text in X:
            feats = {'sel_pos': 0, 'sel_neg': 0, 'liwc_pos': 0, 'liwc_neg': 0, 'emoji_pos': 0, 'emoji_neg': 0}
            words = text.split()
            doc_len = len(words) if len(words) > 0 else 1
            
            for word in words:
                clean_word = word.replace("NO_", "")
                is_negated = word.startswith("NO_")

                # Lógica SEL
                if clean_word in self.sel_dict:
                    pol = self.sel_dict[clean_word]
                    # XOR lógico: si es positivo y está negado -> cuenta como negativo
                    if (pol == 'pos') ^ is_negated: feats['sel_pos'] += 1
                    else: feats['sel_neg'] += 1
                
                # Lógica LIWC
                if clean_word in self.liwc_pos:
                    if is_negated: feats['liwc_neg'] += 1
                    else: feats['liwc_pos'] += 1
                elif clean_word in self.liwc_neg:
                    if is_negated: feats['liwc_pos'] += 1
                    else: feats['liwc_neg'] += 1
                
                # Lógica Emojis (Directa)
                if word in self.emoji_scores:
                    vals = self.emoji_scores[word]
                    feats['emoji_pos'] += vals['pos']
                    feats['emoji_neg'] += vals['neg']

            features.append([
                feats['sel_pos']/doc_len, feats['sel_neg']/doc_len,
                feats['liwc_pos']/doc_len, feats['liwc_neg']/doc_len,
                feats['emoji_pos'], feats['emoji_neg']
            ])
        return np.array(features)

# --- FUNCIÓN PÚBLICA PARA EXPORTAR ---
def get_augmented_features():
    """
    Retorna el FeatureUnion configurado con:
    1. Bag of Words (Binary)
    2. Lexicon Features (SEL, LIWC, Emojis)
    """
    return FeatureUnion([
        ('bow', CountVectorizer(binary=True)), # La parte textual ganadora
        ('lexicons', LexiconFeatureExtractor()) # La parte de léxicos
    ])