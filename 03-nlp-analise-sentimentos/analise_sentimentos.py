import pandas as pd
import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nome_do_arquivo = 'Tweets.csv'
resenha = pd.read_csv(nome_do_arquivo)

# Selecionar colunas e filtrar para classificação binária
resenha = resenha[['text', 'airline_sentiment']]
resenha = resenha[resenha['airline_sentiment'] != 'neutral'].copy()

sentiment_map = {'negative': 0, 'positive': 1}
resenha['sentiment'] = resenha['airline_sentiment'].map(sentiment_map)

# Renomear a coluna de texto para corresponder ao código original
resenha.rename(columns={'text': 'review'}, inplace=True)

# Download das stopwords
try:
    palavras_irrelevantes = nltk.corpus.stopwords.words("english")
except LookupError:
    nltk.download('stopwords')
    palavras_irrelevantes = nltk.corpus.stopwords.words("english")

# Tokenizador que mantém apenas palavras
token_espaco = RegexpTokenizer(r'\w+')

# Processamento para remover stopwords
frases_processadas = list()
for opiniao in resenha['review'].astype(str):
  nova_frase = list()
  palavras_texto = token_espaco.tokenize(opiniao.lower())
  for palavra in palavras_texto:
    if palavra not in palavras_irrelevantes:
      nova_frase.append(palavra)
  frases_processadas.append(' '.join(nova_frase))

resenha['tratamento_1'] = frases_processadas

# Vetorização com TF-IDF
tfidf = TfidfVectorizer(lowercase=False)
vetor_tfidf = tfidf.fit_transform(resenha["tratamento_1"])

# Divisão em treino e teste
treino, teste, classe_treino, classe_teste = train_test_split(
    vetor_tfidf,
    resenha.sentiment,
    test_size = 0.3,
    random_state=42
)

# Treinamento do modelo
regressao_logistica = LogisticRegression(max_iter=1000)
regressao_logistica.fit(treino, classe_treino)

# Previsões
y_test_predictions = regressao_logistica.predict(teste)

# Métricas
precision = precision_score(classe_teste, y_test_predictions)
recall = recall_score(classe_teste, y_test_predictions)
f1score = f1_score(classe_teste, y_test_predictions)
acuracia = accuracy_score(classe_teste, y_test_predictions)

# Exibir resultados
print("Resultados do Modelo de Regressão Logística:")
print(f"Acurácia: {acuracia:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1score:.4f}")