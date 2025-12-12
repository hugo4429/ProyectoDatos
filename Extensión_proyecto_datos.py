import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

plt.style.use('ggplot')

print("--- INICIANDO EXTENSIÓN: CLUSTERING TEMÁTICO ---")

try:
    df = pd.read_csv("dataset_procesado_final.csv").dropna()
    print(f"Datos cargados: {len(df)} textos.")
except FileNotFoundError:
    print("Error: No se encuentra el CSV.")
    exit()

print("Generando embeddings semánticos (all-MiniLM-L6-v2)...")
model_emb = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model_emb.encode(df['input_text'].tolist(), show_progress_bar=True)

print("Calculando proyección 2D (t-SNE)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
vis_dims = tsne.fit_transform(embeddings)

df['x'] = vis_dims[:, 0]
df['y'] = vis_dims[:, 1]

NUM_TOPICS = 5
print(f"Agrupando noticias en {NUM_TOPICS} temas principales...")
kmeans = KMeans(n_clusters=NUM_TOPICS, random_state=42, n_init=10)
kmeans.fit(embeddings)
df['cluster'] = kmeans.labels_

print("\n--- TEMAS DETECTADOS (TOP WORDS) ---")
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

def get_top_keywords(data, clusters, labels, n_terms=5):
    df_group = pd.DataFrame(data).groupby(clusters).agg(lambda x: ' '.join(x)).reset_index()
    tf_matrix = tfidf.fit_transform(df_group[data.name])
    feature_names = np.array(tfidf.get_feature_names_out())

    print(f"{'Cluster':<10} | {'Palabras Clave (Temática)'}")
    print("-" * 60)

    topics_summary = {}

    for i in range(len(df_group)):
        top_ids = np.argsort(tf_matrix[i].toarray()[0])[::-1][:n_terms]
        keywords = ", ".join(feature_names[top_ids])
        print(f"Grupo {i:<4} | {keywords}")
        topics_summary[i] = keywords
    return topics_summary

topics_map = get_top_keywords(df['input_text'], df['cluster'], df['label'])

df['Etiqueta'] = df['label'].map({0: 'Neutro', 1: 'Hiperpartidista'})

plt.figure(figsize=(12, 8))

sns.scatterplot(
    data=df,
    x='x', y='y',
    hue='cluster',
    style='Etiqueta',
    palette='tab10',
    s=100,
    alpha=0.8
)

for i in range(NUM_TOPICS):
    cluster_data = df[df['cluster'] == i]
    center_x = cluster_data['x'].mean()
    center_y = cluster_data['y'].mean()
    top_2_words = " ".join(topics_map[i].split(",")[:2])
    plt.text(center_x, center_y, f"G{i}\n{top_2_words}",
             fontsize=10, weight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.title('Mapa Semántico de Noticias: Temas vs. Partidismo', fontsize=15)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.tight_layout()
plt.savefig('mapa_clustering_topicos.png')
print("\n Mapa guardado como 'mapa_clustering_topicos.png'")
plt.show()

crosstab = pd.crosstab(df['cluster'], df['Etiqueta'], normalize='index') * 100
print("\n--- ¿CÓMO DE 'TÓXICO' ES CADA TEMA? ---")
print(crosstab.round(1))