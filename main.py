from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Importa bibliotecas:
# - `CountVectorizer`: converte textos em uma matriz de contagens de palavras.
# - `train_test_split`: divide os dados em conjuntos de treino e teste.
# - `MultinomialNB`: implementa o algoritmo Naive Bayes para classificação.
# - `accuracy_score`: calcula a precisão do modelo.

# Dados de exemplo
textos = [
    "O novo lançamento da Apple",
    "Resultado do jogo de ontem",
    "Eleições presidenciais",
    "Atualização no mundo da tecnologia",
    "Campeonato de futebol",
    "Política internacional"
]
categorias = ["tecnologia", "esportes", "política", "tecnologia", "esportes", "política"]

# `textos` contém frases curtas que serão usadas como dados de entrada.
# `categorias` contém as categorias correspondentes a cada frase.

# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# `CountVectorizer` transforma os textos em uma matriz numérica onde cada linha
# representa um texto e cada coluna representa uma palavra única (token).
# O valor na matriz indica quantas vezes a palavra aparece no texto.

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)

# Divide os dados em dois conjuntos:
# - `X_train` e `y_train`: usados para treinar o modelo.
# - `X_test` e `y_test`: usados para testar o modelo.
# O parâmetro `test_size=0.5` indica que 50% dos dados serão usados para teste.

# Treinando o classificador
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Cria um modelo Naive Bayes (`MultinomialNB`) e o treina com os dados de treino (`X_train` e `y_train`).

# Predição e Avaliação
y_pred = clf.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, y_pred)}")

# Usa o modelo treinado para prever as categorias dos textos de teste (`X_test`).
# Compara as previsões (`y_pred`) com as categorias reais (`y_test`) e calcula a precisão.
# A precisão é exibida no console.