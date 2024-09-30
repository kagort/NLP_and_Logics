import spacy
from colorama import Fore, Style, init
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Убедимся, что стоп-слова скачаны (если это не сделано ранее)
nltk.download('stopwords')

# Инициализация colorama
init(autoreset=True)

# Загрузка модели SpaCy
nlp = spacy.load("ru_core_news_sm")

# Открытие и чтение файла
with open('text1.txt', "r", encoding="utf-16") as f:
    text = f.read().lower()

doc = nlp(text)

# Вводим список ключевых слов, удаляем пробелы по краям
words = [word.strip() for word in input('Введите ключевые слова через запятую:').lower().split(',')]

# Создаем словари для хранения зависимых слов для каждого ключевого слова
Sub = {word: [] for word in words}
Obj = {word: [] for word in words}
Adj = {word: [] for word in words}
Noun = {word: [] for word in words}
Participle = {word: [] for word in words}
word_counts = {word: 0 for word in words}

# Анализ зависимостей
for word in words:
    for sent in doc.sents:
        for token in sent:
            if token.lemma_ == word:
                word_counts[word] += 1
                if token.dep_ == 'nsubj':
                    Sub[word].append(token.head.text)
                if token.dep_ in ('dobj', 'obj', 'obl'):
                    Obj[word].append(token.head.text)
                for ch in token.children:
                    if ch.pos_ == 'ADJ':
                        Adj[word].append(ch.lemma_)
                    elif ch.pos_ == 'NOUN':
                        Noun[word].append(ch.text)
                    elif ch.pos_ == 'VERB':
                        Participle[word].append(ch.text)

# Лемматизация и фильтрация
stop_w = stopwords.words("russian")
lemmatized_Sub = {word: [" ".join([token.lemma_ for token in nlp(text)]) for text in Sub[word]] for word in words}
lemmatized_Obj = {word: [" ".join([token.lemma_ for token in nlp(text)]) for text in Obj[word]] for word in words}
filt_Sub = {word: [w for w in lemmatized_Sub[word] if w not in stop_w] for word in words}
filt_Obj = {word: [w for w in lemmatized_Obj[word] if w not in stop_w] for word in words}
filt_Adj = {word: [w for w in Adj[word] if w not in stop_w] for word in words}
filt_Noun = {word: [w for w in Noun[word] if w not in stop_w] for word in words}
filt_Participle = {word: [w for w in Participle[word] if w not in stop_w] for word in words}

# Частотные слова для контекстных окон
fdist_sw_Sub = {word: FreqDist(filt_Sub[word]) for word in words}
fdist_sw_Obj = {word: FreqDist(filt_Obj[word]) for word in words}
fdist_sw_Adj = {word: FreqDist(filt_Adj[word]) for word in words}
fdist_sw_Noun = {word: FreqDist(filt_Noun[word]) for word in words}
fdist_sw_Participle = {word: FreqDist(filt_Participle[word]) for word in words}

# Самые частотные слова для контекстных окон
tooltips = {
    'Глаголы подлежащего': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Sub[word].most_common(7)]) for word in words],
    'Глаголы дополнения': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Obj[word].most_common(7)]) for word in words],
    'Прилагательные': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Adj[word].most_common(7)]) for word in words],
    'Существительные': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Noun[word].most_common(7)]) for word in words],
    'Причастия': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Participle[word].most_common(7)]) for word in words],
}

# Dash приложение
app = dash.Dash(__name__)

# Создаем график с Plotly
def create_figure():
    # Обновляем ключевые слова с частотой
    words_with_counts = [f"{word} [{word_counts[word]}]" for word in words]

    fig = go.Figure()

    # Глаголы подлежащего
    fig.add_trace(go.Bar(
        x=words_with_counts,
        y=[len(Sub[word]) for word in words],
        name='Глаголы подлежащего',
        marker_color='green',
        customdata=[filt_Sub[word] for word in words],
        hovertext=tooltips['Глаголы подлежащего'],
        hovertemplate='%{hovertext}<extra></extra>',
        text=[len(Sub[word]) for word in words],  # Число над столбцом
        textposition='auto'
    ))

    # Глаголы дополнения
    fig.add_trace(go.Bar(
        x=words_with_counts,
        y=[len(Obj[word]) for word in words],
        name='Глаголы дополнения',
        marker_color='cyan',
        customdata=[filt_Obj[word] for word in words],
        hovertext=tooltips['Глаголы дополнения'],
        hovertemplate='%{hovertext}<extra></extra>',
        text=[len(Obj[word]) for word in words],  # Число над столбцом
        textposition='auto'
    ))

    # Прилагательные
    fig.add_trace(go.Bar(
        x=words_with_counts,
        y=[len(Adj[word]) for word in words],
        name='Прилагательные',
        marker_color='magenta',
        customdata=[filt_Adj[word] for word in words],
        hovertext=tooltips['Прилагательные'],
        hovertemplate='%{hovertext}<extra></extra>',
        text=[len(Adj[word]) for word in words],  # Число над столбцом
        textposition='auto'
    ))

    # Существительные
    fig.add_trace(go.Bar(
        x=words_with_counts,
        y=[len(Noun[word]) for word in words],
        name='Существительные',
        marker_color='red',
        customdata=[filt_Noun[word] for word in words],
        hovertext=tooltips['Существительные'],
        hovertemplate='%{hovertext}<extra></extra>',
        text=[len(Noun[word]) for word in words],  # Число над столбцом
        textposition='auto'
    ))

    # Причастия
    fig.add_trace(go.Bar(
        x=words_with_counts,
        y=[len(Participle[word]) for word in words],
        name='Причастия',
        marker_color='blue',
        customdata=[filt_Participle[word] for word in words],
        hovertext=tooltips['Причастия'],
        hovertemplate='%{hovertext}<extra></extra>',
        text=[len(Participle[word]) for word in words],  # Число над столбцом
        textposition='auto'
    ))

    fig.update_layout(
        title='Интерактивная визуализация зависимых слов по ключевым терминам',
        xaxis_title='Ключевые слова',
        yaxis_title='Количество',
        barmode='group',
        hovermode='x unified'
    )

    return fig

# Layout приложения
app.layout = html.Div([
    dcc.Graph(
        id='interactive-graph',
        figure=create_figure()
    ),
    html.Div(id='output-text')
])

# Обработка клика по графику
@app.callback(
    Output('output-text', 'children'),
    [Input('interactive-graph', 'clickData')]
)
def display_click_data(clickData):
    if clickData is None:
        return "Кликните на столбец, чтобы увидеть список зависимых слов."

    # Получаем данные клика
    point = clickData['points'][0]
    word = point['x'].split(' [')[0]  # Извлекаем только ключевое слово без частоты
    category = point['data']['name']  # Категория

    # Определение категории и соответствующих данных
    category_map = {
        'Глаголы подлежащего': filt_Sub,
        'Глаголы дополнения': filt_Obj,
        'Прилагательные': filt_Adj,
        'Существительные': filt_Noun,
        'Причастия': filt_Participle
    }

    data = category_map.get(category, {}).get(word, [])

    # Возвращаем список зависимых слов
    if data:
        return f"Категория: {category}, Ключевое слово: {word}, Список слов: {', '.join(data)}"
    else:
        return "Нет данных для выбранного элемента."

if __name__ == '__main__':
    app.run_server(debug=True)
