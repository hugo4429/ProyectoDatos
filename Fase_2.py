import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
import nltk
from graphviz import Digraph  
import random
import copy

# ==============================================================================
# CONFIGURACIÓN DE SEMILLA (REPRODUCIBILIDAD)
# ==============================================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"--- Semilla global establecida en: {SEED} ---")

# Descargas de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ==============================================================================
# 1. FUNCIONES AUXILIARES DE EVALUACIÓN Y GRÁFICOS
# ==============================================================================
def evaluar_y_graficar(y_true, y_pred_proba, nombre_modelo):
    """
    Genera matriz de confusión y curva ROC, guarda las imágenes y devuelve métricas.
    """
    y_pred = (y_pred_proba > 0.5).astype(int)

    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)

    # 2. Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Neutro', 'Hiper'], yticklabels=['Neutro', 'Hiper'])
    plt.title(f'Matriz de Confusión: {nombre_modelo}')
    plt.ylabel('Realidad')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(f'conf_matrix_{nombre_modelo.replace(" ", "_")}.png')
    plt.close()

    # 3. Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - {nombre_modelo}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_{nombre_modelo.replace(" ", "_")}.png')
    plt.close()

    return acc, roc_auc

def dibujar_arquitectura_conceptual():
    """Genera un diagrama simplificado y limpio de la red neuronal"""
    try:
        dot = Digraph(comment='Arquitectura W2V Simplificada')
        dot.attr(rankdir='LR', size='10') 

        # DEFINICIÓN DE NODOS
        dot.node('Input', 'Input Layer\n(Vector Google News)\nDimensión: 300',
                 shape='component', style='filled', fillcolor='#E1F5FE')
        dot.node('Hidden1', 'Capa Oculta 1\n(Linear + ReLU)\nNeuronas: 128',
                 shape='box', style='filled', fillcolor='#FFF9C4')
        dot.node('Dropout', 'Dropout\n(0.3)',
                 shape='ellipse', style='dashed', color='gray')
        dot.node('Hidden2', 'Capa Oculta 2\n(Linear + ReLU)\nNeuronas: 64',
                 shape='box', style='filled', fillcolor='#FFF9C4')
        dot.node('Output', 'Capa de Salida\n(Sigmoid)\nNeuronas: 1',
                 shape='circle', style='filled', fillcolor='#C8E6C9')
        dot.node('Result', 'Probabilidad\n(Hiperpartidista)', shape='plaintext')

        # CONEXIONES
        dot.edge('Input', 'Hidden1', label=' features')
        dot.edge('Hidden1', 'Dropout')
        dot.edge('Dropout', 'Hidden2')
        dot.edge('Hidden2', 'Output')
        dot.edge('Output', 'Result')

        output_path = 'diagrama_w2v_simple'
        dot.render(output_path, format='png', cleanup=True)
        print(f"Diagrama simplificado guardado como '{output_path}.png'")
    except Exception as e:
        print(f"No se pudo generar el diagrama (revisa si graphviz está instalado): {e}")

# ==============================================================================
# 2. CLASE RED NEURONAL GENÉRICA
# ==============================================================================
class GenericNN(nn.Module):
    def __init__(self, input_dim):
        super(GenericNN, self).__init__()
        torch.manual_seed(SEED)
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return self.sigmoid(out)

# FUNCIÓN DE ENTRENAMIENTO CON EARLY STOPPING
def entrenar_pytorch(X_train, y_train, X_val, y_val, X_test, y_test, input_dim, nombre, epochs=200, patience=5):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(SEED)
    model = GenericNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Entrenando NN ({nombre}) con Early Stopping")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss_train = criterion(outputs, y_train_t)
        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            loss_val = criterion(val_outputs, y_val_t)

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"      Epoca {epoch+1}/{epochs} | Train Loss: {loss_train:.4f} | Val Loss: {loss_val:.4f}")

        if patience_counter >= patience:
            print(f"Early Stopping activado en época {epoch+1}. Recuperando el mejor modelo.")
            break

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    model.eval()
    with torch.no_grad():
        y_probs = model(X_test_t).numpy().flatten()

    return y_probs

# ==============================================================================
# 3. PREPARACIÓN DE DATOS
# ==============================================================================
print("--- CARGANDO DATOS ---")
df = pd.read_csv("dataset_procesado_final.csv").dropna()
X_text = df['input_text']
y = df['label']

X_train_raw, X_temp, y_train, y_temp = train_test_split(X_text, y, test_size=0.3, random_state=SEED, stratify=y)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

resultados = []

# ==============================================================================
# 4. TÉCNICA 1: TF-IDF (Baseline)
# ==============================================================================
print("\n--- PROCESANDO 1: TF-IDF ---")
tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=10)
X_train_tfidf = tfidf.fit_transform(X_train_raw).toarray()
X_val_tfidf = tfidf.transform(X_val_raw).toarray()
X_test_tfidf = tfidf.transform(X_test_raw).toarray()

# A. Sklearn
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr.fit(X_train_tfidf, y_train)
probs = lr.predict_proba(X_test_tfidf)[:, 1]
acc, auc_score = evaluar_y_graficar(y_test, probs, "TFIDF_Sklearn")
resultados.append(["TFIDF + Sklearn", acc, auc_score])

# B. PyTorch
probs = entrenar_pytorch(X_train_tfidf, y_train, X_val_tfidf, y_val, X_test_tfidf, y_test,
                         X_train_tfidf.shape[1], "TFIDF_NN", epochs=200, patience=5)
acc, auc_score = evaluar_y_graficar(y_test, probs, "TFIDF_PyTorch")
resultados.append(["TFIDF + PyTorch", acc, auc_score])

# ==============================================================================
# 5. TÉCNICA 2: WORD2VEC (GOOGLE NEWS PRE-TRAINED)
# ==============================================================================
print("\n--- PROCESANDO 2: WORD2VEC (GOOGLE NEWS) ---")
print("Descargando/Cargando modelo Google News")
w2v_model = api.load('word2vec-google-news-300')

def document_vector(doc, model):
    tokens = nltk.word_tokenize(doc.lower())
    vectors = [model[w] for w in tokens if w in model]
    if len(vectors) == 0:
        return np.zeros(300)
    return np.mean(vectors, axis=0)

print("Vectorizando textos con Google W2V")
X_train_w2v = np.array([document_vector(d, w2v_model) for d in X_train_raw])
X_val_w2v = np.array([document_vector(d, w2v_model) for d in X_val_raw])
X_test_w2v = np.array([document_vector(d, w2v_model) for d in X_test_raw])

# A. Sklearn
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr.fit(X_train_w2v, y_train)
probs = lr.predict_proba(X_test_w2v)[:, 1]
acc, auc_score = evaluar_y_graficar(y_test, probs, "W2V_Google_Sklearn")
resultados.append(["W2V_Google + Sklearn", acc, auc_score])

# B. PyTorch
probs = entrenar_pytorch(X_train_w2v, y_train, X_val_w2v, y_val, X_test_w2v, y_test,
                         300, "W2V_Google_NN", epochs=200, patience=5)
acc, auc_score = evaluar_y_graficar(y_test, probs, "W2V_Google_PyTorch")
resultados.append(["W2V_Google + PyTorch", acc, auc_score])

# --- DIAGRAMA W2V SIMPLIFICADO ---
print("Generando diagrama conceptual simplificado de la red W2V")
dibujar_arquitectura_conceptual()
# --------------------

# ==============================================================================
# 6. TÉCNICA 3: BERT EMBEDDINGS
# ==============================================================================
print("\n--- PROCESANDO 3: BERT EMBEDDINGS ---")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_bert_embeddings(text_list):
    print(f"Extrayendo embeddings para {len(text_list)} textos")
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state[0][0].numpy())
    return np.array(embeddings)

X_train_bert = get_bert_embeddings(X_train_raw.tolist())
X_val_bert = get_bert_embeddings(X_val_raw.tolist())
X_test_bert = get_bert_embeddings(X_test_raw.tolist())

# A. Sklearn
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
lr.fit(X_train_bert, y_train)
probs = lr.predict_proba(X_test_bert)[:, 1]
acc, auc_score = evaluar_y_graficar(y_test, probs, "BERT_Emb_Sklearn")
resultados.append(["BERT_Emb + Sklearn", acc, auc_score])

# B. PyTorch
probs = entrenar_pytorch(X_train_bert, y_train, X_val_bert, y_val, X_test_bert, y_test,
                         768, "BERT_Emb_NN", epochs=200, patience=5)
acc, auc_score = evaluar_y_graficar(y_test, probs, "BERT_Emb_PyTorch")
resultados.append(["BERT_Emb + PyTorch", acc, auc_score])

# ==============================================================================
# RESUMEN FINAL
# ==============================================================================
print("\n" + "="*50)
print(f"{'Modelo':<25} | {'Accuracy':<10} | {'AUC-ROC':<10}")
print("-" * 50)
for fila in resultados:
    print(f"{fila[0]:<25} | {fila[1]:.4f}     | {fila[2]:.4f}")
print("-" * 50)
print("Imágenes guardadas en la carpeta actual")
