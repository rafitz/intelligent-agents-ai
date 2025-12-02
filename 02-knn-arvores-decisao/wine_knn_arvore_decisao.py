import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

"""# Carrergando Dataset"""

# Carregar o dataset Wine
# O dataset contém informações químicas de diferentes vinhos classificados em 3 categorias
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="wine_class")

"""# Dividir os dados em treino e teste (o nome dessa divisão é houldout)"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

"""# Normalizar os dados"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""# Escolher o melhor K"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_values = range(1, 21)  # vamos testar K de 1 até 20
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

best_k = k_values[accuracies.index(max(accuracies))]
best_acc = max(accuracies)

print(f"Melhor K: {best_k} com acurácia de {best_acc:.2f}")

import matplotlib.pyplot as plt

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Número de Vizinhos (K)")
plt.ylabel("Acurácia")
plt.title("Escolha do melhor K")
plt.show()

"""# Treinar Algoritmo com Melhor K"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Acurácia final no teste: {accuracy_score(y_test, y_pred):.2f}")

"""# Classificar os dados de teste e exibir a acurácia"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Treinar modelo com o melhor K
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# Classificar os dados de teste
y_pred = knn.predict(X_test_scaled)

# Exibir acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia final no conjunto de teste: {accuracy:.2f}")

"""## **ARVORE DE DECISÃO**"""

# Importações de bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import gradio as gr

# 1. Carregamento e Pré-processamento dos Dados
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)
target_names = wine.target_names

# Normalização dos dados (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Definição dos Classificadores
knn_model = KNeighborsClassifier(n_neighbors=13)
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

# 3. Configuração da Validação Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4. Função para avaliar modelos
def evaluate_model(model, model_name, X_data, y_data, cv_strategy):
    print(f"\\n--- Avaliação do Modelo: {model_name} ---")

    scoring_metrics = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

    scores = cross_validate(model, X_data, y_data, cv=cv_strategy, scoring=scoring_metrics)

    print(f"Acurácia Média:     {np.mean(scores['test_accuracy']):.4f}")
    print(f"Precisão Média:     {np.mean(scores['test_precision_macro']):.4f}")
    print(f"Recall Médio:       {np.mean(scores['test_recall_macro']):.4f}")
    print(f"F1-Score Médio:     {np.mean(scores['test_f1_macro']):.4f}")

    y_pred = cross_val_predict(model, X_data, y_data, cv=cv_strategy)

    print("\\nRelatório de Classificação:")
    print(classification_report(y_data, y_pred, target_names=target_names))

    cm = confusion_matrix(y_data, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()

evaluate_model(knn_model, "KNN (k=13)", X_scaled, y, cv)
evaluate_model(dt_model, "Árvore de Decisão (Entropia)", X_scaled, y, cv)

knn_model.fit(X_scaled, y)
dt_model.fit(X_scaled, y)

def predict_wine_class(*features):
    input_array = np.array(features).reshape(1, -1)

    input_scaled = scaler.transform(input_array)

    pred_knn = knn_model.predict(input_scaled)[0]
    pred_dt = dt_model.predict(input_scaled)[0]

    return target_names[pred_knn], target_names[pred_dt]

with gr.Blocks() as demo:
    gr.Markdown("## 🍷 Classificador de Vinhos: KNN vs. Árvore de Decisão")
    gr.Markdown("Ajuste os valores das 13 características abaixo e clique em 'Prever' para ver a classificação de cada modelo.")

    with gr.Row():
        with gr.Column():
            inputs_col1 = [gr.Slider(minimum=X[col].min(), maximum=X[col].max(), value=X[col].mean(), label=col) for col in X.columns[:7]]
        with gr.Column():
            inputs_col2 = [gr.Slider(minimum=X[col].min(), maximum=X[col].max(), value=X[col].mean(), label=col) for col in X.columns[7:]]

    all_inputs = inputs_col1 + inputs_col2

    with gr.Row():
        output_knn = gr.Textbox(label="Predição do KNN (k=13)")
        output_dt = gr.Textbox(label="Predição da Árvore de Decisão")

    btn = gr.Button("Prever")
    btn.click(fn=predict_wine_class, inputs=all_inputs, outputs=[output_knn, output_dt])

demo.launch()