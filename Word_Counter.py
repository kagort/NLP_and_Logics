import spacy
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Убедимся, что стоп-слова скачаны
nltk.download('stopwords')

# Загрузка модели SpaCy
nlp = spacy.load("ru_core_news_sm")

# Открытие и чтение файла
with open('text3N.txt', "r", encoding="utf-16") as f:
    text = f.read().lower()

# Обрабатываем текст с помощью SpaCy
doc = nlp(text)

# Лемматизация слов и фильтрация по стоп-словам
words = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords.words("russian")]

# Частотный анализ лемм
fdist = FreqDist(words)
most_common_words = fdist.most_common(100)  # Топ-100 самых частых слов

# Функция для группировки слов по частоте
def group_words_by_frequency(common_words):
    grouped_words = {}
    for word, count in common_words:
        if count not in grouped_words:
            grouped_words[count] = []
        grouped_words[count].append(word)
    return grouped_words

# Создание Dash приложения
app = dash.Dash(__name__)

# Layout приложения
app.layout = html.Div([
    html.H1("Топ-100 самых частых слов в тексте"),

    # Хранилище для исключений
    dcc.Store(id='excluded-words', data=[]),

    # Отображение списка самых частых слов
    html.Div(id='word-list-container', children=[
        html.Ol(id='word-list')  # Нумерованный список
    ]),

    # Список исключенных слов
    html.H2("Исключенные слова"),
    html.Div(id='excluded-word-list-container', children=[
        html.Ul(id='excluded-word-list')
    ])
])

# Callback для обновления списка исключенных слов и топ-100
@app.callback(
    [Output('excluded-word-list', 'children'),  # Обновляем отображение исключенных слов
     Output('word-list', 'children'),  # Обновляем отображение списка самых частых слов
     Output('excluded-words', 'data')],  # Обновляем состояние исключенных слов
    [Input({'type': 'word-item', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('excluded-words', 'data')]  # Получаем текущее состояние исключенных слов
)
def update_excluded_word_list(n_clicks, excluded_words):
    # Определяем, какое слово было кликнуто
    ctx = dash.callback_context
    if not ctx.triggered:
        # Если ничего не было нажато, возвращаем первоначальные значения
        grouped_words = group_words_by_frequency(most_common_words)
        return [], create_numbered_word_list(grouped_words, excluded_words), excluded_words

    triggered_word = ctx.triggered[0]['prop_id'].split('.')[0]
    triggered_word = eval(triggered_word)['index']

    # Добавляем кликнутое слово в список исключенных
    if triggered_word not in excluded_words:
        excluded_words.append(triggered_word)

    # Обновляем список самых частых слов, исключая слова
    updated_common_words = [(word, count) for word, count in fdist.most_common(100) if word not in excluded_words]
    grouped_words = group_words_by_frequency(updated_common_words)

    # Обновляем HTML для списков
    updated_word_list = create_numbered_word_list(grouped_words, excluded_words)
    excluded_word_list = [
        html.Li(word) for word in excluded_words
    ]

    # Возвращаем обновленные списки и состояние исключенных слов
    return excluded_word_list, updated_word_list, excluded_words

# Функция для создания нумерованного списка слов
def create_numbered_word_list(grouped_words, excluded_words):
    """
    Создает нумерованный список слов с группировкой по частоте.
    """
    word_list = []
    rank = 1  # Начальный порядковый номер

    for count, words in grouped_words.items():
        words_in_group = [word for word in words if word not in excluded_words]  # Убираем исключенные слова
        if words_in_group:  # Если группа не пуста
            word_list.append(html.Li(f"Частота {count}: " + ', '.join(words_in_group), id={'type': 'word-item', 'index': ','.join(words_in_group)}))
            rank += 1

    return word_list

# Запуск приложения
if __name__ == '__main__':
    app.run_server(debug=True)
