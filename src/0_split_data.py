import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuración de Rutas (Robusta) ---
# Obtiene la ruta absoluta del script actual (ej. .../src/0_split_data.py)
SCRIPT_PATH = os.path.abspath(__file__)
# Sube un nivel al directorio 'src'
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
# Sube otro nivel al directorio raíz del proyecto (ej. .../sentimental-analysis)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) 

# Ahora construye las rutas desde la raíz del proyecto
RAW_CORPUS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "Rest_Mex_2022.xlsx")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_PATH = os.path.join(PROCESSED_PATH, "train.csv")
TEST_PATH = os.path.join(PROCESSED_PATH, "test.csv")

def main():
    print("Iniciando Fase 0: División de Datos")
    
    # Asegurarse que la carpeta de destino exista
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    try:
        # 1. Cargar el corpus
        df = pd.read_excel(RAW_CORPUS_PATH)
        print(f"Corpus crudo cargado: {len(df)} filas.")

        # 2. Concatenar Title y Opinion 
        df['Title'] = df['Title'].fillna('').astype(str)
        df['Opinion'] = df['Opinion'].fillna('').astype(str)

        df['Title'] = df['Title'].replace('nan', '')
        df['Opinion'] = df['Opinion'].replace('nan', '')

        df['text'] = df['Title'].str.strip() + ' ' + df['Opinion'].str.strip()
        df['text'] = df['text'].str.strip()
        
        # 3. Definir X (features) e y (target)
        # Elimina filas donde el texto o la polaridad sean nulos
        df.dropna(subset=['text', 'Polarity'], inplace=True)
        # Filtra textos vacíos que pudieron quedar
        df = df[df['text'] != '']
        
        X = df['text'] # features
        y = df['Polarity'] # class
        
        print(f"Datos limpios listos para dividir: {len(X)} filas.")

        # 4. Crear la división 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.20,
            random_state=0,      # Fijo para reproducibilidad
            shuffle=True         # Mezclar los datos
        )
        
        print(f"Tamaño de Entrenamiento: {len(X_train)} (80%)")
        print(f"Tamaño de Prueba: {len(X_test)} (20%)")

        # 5. Guardar los sets en archivos separados
        train_df = pd.DataFrame({'text': X_train, 'Polarity': y_train})
        test_df = pd.DataFrame({'text': X_test, 'Polarity': y_test})
        
        train_df.to_csv(TRAIN_PATH, index=False, encoding='utf-8')
        test_df.to_csv(TEST_PATH, index=False, encoding='utf-8')
        
        print("\n¡Éxito! Archivos 'train.csv' y 'test.csv' guardados en 'data/processed/'.")
        print("Ahora todo el equipo puede usar estos archivos.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {RAW_CORPUS_PATH}")
        print(f"Asegúrate de que el corpus '{os.path.basename(RAW_CORPUS_PATH)}' esté en la carpeta 'data/raw/'.")

if __name__ == "__main__":
    main()