import os
# Desactivamos W&B para que no pida login
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

# ==============================================================================
# CONFIGURACIÓN INICIAL Y SEMILLAS
# ==============================================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("--- INICIANDO FINE-TUNING CON RoBERTa-BASE (Versión Final) ---")

# ==============================================================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==============================================================================
df = pd.read_csv("dataset_procesado_final.csv").dropna()

df_trainval, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
train_df, val_df = train_test_split(df_trainval, test_size=0.15, random_state=SEED, stratify=df_trainval["label"])

classes = np.unique(train_df["label"])
class_weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["label"].values)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
print("Pesos de clase aplicados:", class_weights)

hg_train = Dataset.from_pandas(train_df.reset_index(drop=True))
hg_val   = Dataset.from_pandas(val_df.reset_index(drop=True))
hg_test  = Dataset.from_pandas(test_df.reset_index(drop=True))

# ==============================================================================
# 2. TOKENIZACIÓN
# ==============================================================================
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=256)

tokenized_train = hg_train.map(preprocess_function, batched=True)
tokenized_val   = hg_val.map(preprocess_function, batched=True)
tokenized_test  = hg_test.map(preprocess_function, batched=True)

for dataset in [tokenized_train, tokenized_val, tokenized_test]:
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ==============================================================================
# 3. MODELO Y CONFIGURACIÓN DE ENTRENAMIENTO
# ==============================================================================
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

training_args = TrainingArguments(
    output_dir="./resultados_roberta",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=20,             
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,     
    metric_for_best_model="f1",      
    greater_is_better=True,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True,                     
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1": f1}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

print("--- ENTRENANDO MODELO... ---")
trainer.train()

# ==============================================================================
# 4. EVALUACIÓN FINAL (TEST)
# ==============================================================================
print("\n--- EVALUANDO EN TEST (Mejor Modelo) ---")
results = trainer.predict(tokenized_test)

logits_tensor = torch.tensor(results.predictions)
probs = F.softmax(logits_tensor, dim=-1).numpy()
probs_hiper = probs[:, 1]
y_pred = np.argmax(results.predictions, axis=-1)
y_true = results.label_ids

print(classification_report(y_true, y_pred, target_names=["Neutro", "Hiper"]))

# ==============================================================================
# 5. GENERACIÓN DE GRÁFICAS
# ==============================================================================
print("\nGenerando gráficas finales...")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neutro', 'Hiper'], yticklabels=['Neutro', 'Hiper'])
plt.title('Matriz de Confusión: RoBERTa-Base (Final)')
plt.ylabel('Realidad')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('conf_matrix_roberta_final.png')
print("Matriz guardada: conf_matrix_roberta_final.png")
plt.show()

fpr, tpr, _ = roc_curve(y_true, probs_hiper)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - RoBERTa-Base (Final)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_roberta_final.png')
print("Curva ROC guardada: roc_roberta_final.png")
plt.show()