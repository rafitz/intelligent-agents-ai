import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')

dados = pd.read_csv('Consumo_cerveja.csv', sep = ';')
dados.describe()

dados.head()

amostra =  dados.query('consumo<25000').sample(n=20, random_state=101)
amostra

amostra[['temp_media',	'temp_min',	'temp_max',	'chuva',	'fds',	'consumo']].cov()

#Construir a reta de regressão achando Beta 1 e Beta 2
x = dados['temp_max']
y = dados['consumo']

total_pontos = len(dados)
soma_x = x.sum()
soma_y = y.sum()
soma_xy = (x * y).sum()
soma_x_quadrado = (x**2).sum()

numerador = (total_pontos * soma_xy) - (soma_x * soma_y)
denominador = (total_pontos * soma_x_quadrado) - (soma_x**2)

beta2 = numerador / denominador
beta1 = y.mean() - beta2 * x.mean()

print(f"--- Regressão Linear ---")
print(f"β1 (intercepto): {beta1:.2f}")
print(f"β2 (inclinação): {beta2:.2f}")
print("\nEquação da Reta:")
print(f"Consumo = {beta1:.2f} + {beta2:.2f} * (Temperatura Máxima)")

#Qual o valor de R² (R-Square)
X = dados[['temp_max', "temp_media", "fds"]]
y = dados['consumo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred_treino = modelo.predict(X_train)
y_pred_teste = modelo.predict(X_test)

r2_treino = metrics.r2_score(y_train, y_pred_treino)
r2_teste = metrics.r2_score(y_test, y_pred_teste)


print("--- Comparação de R² ---")
print(f"R² Treino: {r2_treino:.4f}")
print(f"R² Teste: {r2_teste:.4f}")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

parametros = {
    'knn__n_neighbors': range(1, 21)
}

grid_search = GridSearchCV(pipeline_knn, parametros, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

melhor_k = grid_search.best_params_['knn__n_neighbors']
r2_teste_knn = grid_search.score(X_test, y_test)
r2_treino_knn = grid_search.score(X_train, y_train)

print("--- Análise com KNN Otimizado ---")
print(f"Melhor número de vizinhos (k) encontrado: {melhor_k}")
print(f"R² Treino: {r2_treino_knn:.4f}")
print(f"R² Teste: {r2_teste_knn:.4f}")

#Construir a reta de regressão achando Beta 1 e Beta 2
x_label = 'Temperatura Máxima'
y_label = 'Consumo de Cerveja'

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.75, label='Dados Observados')

# Linha da regressão (usa os betas calculados)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = beta1 + beta2 * x_line
plt.plot(x_line, y_line, color='red', lw=2, label='Reta de Regressão')

plt.title(f'{y_label} vs {x_label}')
plt.xlabel(x_label)
plt.ylabel(f'{y_label} (litros)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Regressão Linear vs. KNN
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

X = dados[['temp_max']]
y = dados['consumo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_linear = LinearRegression()
modelo_linear.fit(X_train, y_train)
r2_linear = modelo_linear.score(X_test, y_test)


parametros = {'knn__n_neighbors': range(1, 21)}
grid_search_knn = GridSearchCV(pipeline_knn, parametros, cv=5, scoring='r2')
grid_search_knn.fit(X_train, y_train)
r2_knn = grid_search_knn.score(X_test, y_test)
melhor_k = grid_search_knn.best_params_['knn__n_neighbors']

print("--- COMPARAÇÃO FINAL DOS MODELOS ---")
print(f"R² do Modelo de Regressão Linear: {r2_linear:.4f}")
print(f"R² do Modelo KNN Otimizado (com k={melhor_k}): {r2_knn:.4f}")
print("-" * 35)

if r2_knn > r2_linear:
    print("Conclusão: O modelo KNN teve um desempenho melhor.")
else:
    print("Conclusão: O modelo de Regressão Linear teve um desempenho melhor.")