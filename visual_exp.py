import nltk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def load_text(filename):
    """Загружает текст из файла.

    Args:
        filename: Имя файла.

    Returns:
        str: Текст.
    """
    with open('text2.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def preprocess_text(text):
    """Предобрабатывает текст.

    Args:
        text: Текст.

    Returns:
        list: Список токенов.
    """
    # Токенизация
    tokens = nltk.word_tokenize(text)
    # Нормализация (стемминг)
    stemmer = nltk.stem.PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # Удаление стоп-слов
    stop_words = set(nltk.corpus.stopwords.words('russian'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def get_keywords(text, num_keywords=10):
    """Определяет ключевые слова с помощью TF-IDF.

    Args:
        text: Текст.
        num_keywords: Количество ключевых слов.

    Returns:
        list: Список ключевых слов.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = X.toarray().argsort()
    top_n_ids = tfidf_sorting[0, -num_keywords:]
    top_features = feature_array[top_n_ids]
    return list(top_features)

def create_frequency_matrix(text, keywords, window_size=10):
    """Создает матрицу частотности ключевых слов в окнах.

    Args:
        text: Текст.
        keywords: Список ключевых слов.
        window_size: Размер окна.

    Returns:
        pd.DataFrame: Матрица частотности.
    """
    tokens = preprocess_text(text)
    matrix = np.zeros((len(keywords), len(tokens) - window_size + 1))
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i+window_size]
        for j, keyword in enumerate(keywords):
            matrix[j, i] = window.count(keyword)
    return pd.DataFrame(matrix, index=keywords)

def plot_heatmap(matrix):
    """Строит тепловую карту.

    Args:
        matrix: Матрица частотности.
    """
    sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm')
    plt.xlabel('Окна')
    plt.ylabel('Ключевые слова')
    plt.show()

# Основной блок
filename = 'ваш_файл.txt'  # Замените на ваше имя файла
text = load_text(filename)
keywords = get_keywords(text)
matrix = create_frequency_matrix(text, keywords)
plot_heatmap(matrix)