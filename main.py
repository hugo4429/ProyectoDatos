from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# --- 1. CARGA DE DATOS ---
# ==============================================================================
print("Cargando dataset...")

dataset = load_dataset("hyperpartisan_news_detection", "byarticle", trust_remote_code=True)


# ==============================================================================
# --- 2. DEFINICIÓN DE FUNCIONES DE LIMPIEZA ---
# ==============================================================================
def limpiar_fila(fila):
    """
    Elimina etiquetas HTML y limpia espacios en blanco.
    """
    texto_sucio = fila['text']
    if texto_sucio:
        soup = BeautifulSoup(texto_sucio, "html.parser")
        
        texto_limpio = soup.get_text(separator=' ').strip()
    else:
        texto_limpio = ""
    
    return {"text_clean": texto_limpio}


# ==============================================================================
# --- 3. APLICACIÓN DE LIMPIEZA ---
# ==============================================================================
print("Limpiando todo el dataset (esto puede tardar unos segundos)...")
dataset_limpio = dataset.map(limpiar_fila)


df = pd.DataFrame(dataset_limpio['train'])


# ==============================================================================
# --- 4. ANÁLISIS DE LONGITUD (EDA) Y GRÁFICO ---
# ==============================================================================
print("\n--- INICIANDO ANÁLISIS DE LONGITUD ---")


df['num_palabras'] = df['text_clean'].apply(lambda x: len(str(x).split()))


plt.figure(figsize=(10, 6))
plt.hist(df['num_palabras'], bins=50, color='skyblue', edgecolor='black')

plt.axvline(1000, color='red', linestyle='dashed', linewidth=2, label='Límite 1000 palabras')
plt.title('Distribución de longitud de las noticias')
plt.xlabel('Número de palabras')
plt.ylabel('Cantidad de artículos')
plt.legend()


plt.savefig('grafico_longitud.png')
print("Gráfico guardado correctamente como 'grafico_longitud.png'")


# ==============================================================================
# --- 5. RECORTE DE TEXTOS LARGOS (>1000 PALABRAS) ---
# ==============================================================================
print("\n--- APLICANDO RECORTE INTELIGENTE (Head + Tail) ---")

def recortar_texto(texto, limite=1000):
    """
    Si el texto supera el límite, conserva el principio y el final.
    Estrategia: 800 palabras iniciales + 200 finales.
    """
    palabras = texto.split()
    if len(palabras) <= limite:
        return texto
    
    n_inicio = 800
    n_final = 200
    
    parte_inicio = palabras[:n_inicio]
    parte_final = palabras[-n_final:]
  
    return " ".join(parte_inicio) + " [...] " + " ".join(parte_final)


df['text_clean'] = df['text_clean'].apply(lambda x: recortar_texto(str(x)))

max_len_nuevo = df['text_clean'].apply(lambda x: len(str(x).split())).max()
print(f"Nueva longitud máxima en el dataset: {max_len_nuevo} palabras (aprox)")


# ==============================================================================
# --- 6. PREPARACIÓN FINAL Y GUARDADO ---
# ==============================================================================
print("\n--- Preparando dataset final para IA... ---")

df['input_text'] = df['title'] + ". " + df['text_clean']

df['label'] = df['hyperpartisan'].astype(int)

df_final = df[['input_text', 'label']]

nombre_archivo = "dataset_procesado_final.csv"
df_final.to_csv(nombre_archivo, index=False)

print(f"¡Proceso completado! Archivo guardado en: '{nombre_archivo}'")
print(df_final.head())