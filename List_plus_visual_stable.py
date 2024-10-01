import spacy
from colorama import Fore, Style, init
from nltk.corpus import stopwords  # Импортируем стоп-слова
from nltk.probability import FreqDist  # Импортируем FreqDist для частотного анализа
import nltk
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go  # Импортируем plotly для интерактивных графиков

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
Sub = {word: [] for word in words}  # глагол подлежащего для каждого ключевого слова
Obj = {word: [] for word in words}  # глагол дополнения для каждого ключевого слова
Adj = {word: [] for word in words}  # прилагательные для каждого ключевого слова
Noun = {word: [] for word in words}  # существительные для каждого ключевого слова
Participle = {word: [] for word in words}  # причастия для каждого ключевого слова
word_counts = {word: 0 for word in words}  # Счетчик вхождений для каждого поискового термина

# Нумерация предложений
print("Список предложений в тексте:")
for i, sent in enumerate(doc.sents, 1):
    print(f"{i}. {sent}")

print("\nНачало анализа...\n")

# Выборка глаголов, существительных, прилагательных и причастий по каждому поисковому термину
for word in words:
    print(f"\nАнализ для термина: {word}\n")
    for sent in doc.sents:
        print(sent)
        for token in sent:
            if token.lemma_ == word:
                a = token
                word_counts[word] += 1  # Увеличиваем счетчик поискового термина

                if a.dep_ == 'nsubj':  # как подлежащее
                    Sub[word].append(a.head.text)  # Записываем только для текущего ключевого слова
                if a.dep_ in ('dobj', 'obj', 'obl'):  # как дополнение
                    Obj[word].append(a.head.text)  # Записываем только для текущего ключевого слова

                # Зависимые слова
                for ch in a.children:
                    if ch.pos_ == 'ADJ':
                        Adj[word].append(ch.lemma_)
                    elif ch.pos_ == 'NOUN':
                        Noun[word].append(ch.text)
                    elif ch.pos_ == 'VERB':
                        Participle[word].append(ch.text)

# Лемматизация списков Sub и Obj
lemmatized_Sub = {word: [" ".join([token.lemma_ for token in nlp(text)]) for text in Sub[word]] for word in words}
lemmatized_Obj = {word: [" ".join([token.lemma_ for token in nlp(text)]) for text in Obj[word]] for word in words}

# Работа со стоп-словами
stop_w = stopwords.words("russian")
stop_w.extend(['�', '-', '―', '”', 'запах', 'вкус'])

filt_Sub = {word: [w for w in lemmatized_Sub[word] if w not in stop_w] for word in words}
filt_Obj = {word: [w for w in lemmatized_Obj[word] if w not in stop_w] for word in words}
filt_Adj = {word: [w for w in Adj[word] if w not in stop_w] for word in words}
filt_Noun = {word: [w for w in Noun[word] if w not in stop_w] for word in words}
filt_Participle = {word: [w for w in Participle[word] if w not in stop_w] for word in words}

# Удаление повторов и сортировка
sub = {word: sorted(list(set(filt_Sub[word]))) for word in words}
obj = {word: sorted(list(set(filt_Obj[word]))) for word in words}
adj = {word: sorted(list(set(filt_Adj[word]))) for word in words}
noun = {word: sorted(list(set(filt_Noun[word]))) for word in words}
participle = {word: sorted(list(set(filt_Participle[word]))) for word in words}

# Частотные слова для контекстных окон
fdist_sw_Sub = {word: FreqDist(filt_Sub[word]) for word in words}
fdist_sw_Obj = {word: FreqDist(filt_Obj[word]) for word in words}
fdist_sw_Adj = {word: FreqDist(filt_Adj[word]) for word in words}
fdist_sw_Noun = {word: FreqDist(filt_Noun[word]) for word in words}
fdist_sw_Participle = {word: FreqDist(filt_Participle[word]) for word in words}

# Самые частотные слова для контекстных окон
tooltips = {
    'Глаголы подлежащего': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Sub[word].most_common(5)]) for word in
                            words],
    'Глаголы дополнения': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Obj[word].most_common(5)]) for word in
                           words],
    'Прилагательные': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Adj[word].most_common(5)]) for word in words],
    'Существительные': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Noun[word].most_common(5)]) for word in words],
    'Причастия': [', '.join([f'{w[0]} ({w[1]})' for w in fdist_sw_Participle[word].most_common(5)]) for word in words],
}

# Dash приложение
app = dash.Dash(__name__)


# Создаем график с Plotly
# Добавляем частоту ключевых слов в заголовок гистограммы
def create_figure():
    fig = go.Figure()

    # Формируем ключевые слова с указанием частотности
    words_with_freq = [f'{word} ({word_counts[word]})' for word in words]

    # Глаголы подлежащего
    fig.add_trace(go.Bar(
        x=words_with_freq,  # Обновляем ось X
        y=[len(Sub[word]) for word in words],
        name='Глаголы подлежащего',
        marker_color='green',
        customdata=[filt_Sub[word] for word in words],
        hovertext=tooltips['Глаголы подлежащего'],  # Контекстные окна с частотными словами
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    # Глаголы дополнения
    fig.add_trace(go.Bar(
        x=words_with_freq,
        y=[len(Obj[word]) for word in words],
        name='Глаголы дополнения',
        marker_color='cyan',
        customdata=[filt_Obj[word] for word in words],
        hovertext=tooltips['Глаголы дополнения'],
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    # Прилагательные
    fig.add_trace(go.Bar(
        x=words_with_freq,
        y=[len(Adj[word]) for word in words],
        name='Прилагательные',
        marker_color='magenta',
        customdata=[filt_Adj[word] for word in words],
        hovertext=tooltips['Прилагательные'],
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    # Существительные
    fig.add_trace(go.Bar(
        x=words_with_freq,
        y=[len(Noun[word]) for word in words],
        name='Существительные',
        marker_color='red',
        customdata=[filt_Noun[word] for word in words],
        hovertext=tooltips['Существительные'],
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    # Причастия
    fig.add_trace(go.Bar(
        x=words_with_freq,
        y=[len(Participle[word]) for word in words],
        name='Причастия',
        marker_color='blue',
        customdata=[filt_Participle[word] for word in words],
        hovertext=tooltips['Причастия'],
        hovertemplate='%{hovertext}<extra></extra>'
    ))

    fig.update_layout(
        title='Интерактивная визуализация зависимых слов по ключевым терминам',
        xaxis_title='Ключевые слова (с частотой)',
        yaxis_title='Количество',
        barmode='group',
        hovermode='x unified'
    )

    return fig

# Обработка клика и вывод слов с их частотностью в порядке убывания
@app.callback(
    Output('output-text', 'children'),
    [Input('interactive-graph', 'clickData')]
)
def display_click_data(clickData):
    if clickData is None:
        return "Кликните на столбец, чтобы увидеть список зависимых слов."

    # Получаем категорию и ключевое слово
    point = clickData['points'][0]
    label_with_freq = point['label']  # Ключевое слово с частотой
    label = label_with_freq.split(' (')[0]  # Извлекаем само ключевое слово
    category = point['curveNumber']  # Категория

    categories = ['Глаголы подлежащего', 'Глаголы дополнения', 'Прилагательные', 'Существительные', 'Причастия']

    # Определяем, какая категория была кликнута
    if category == 0:
        data = fdist_sw_Sub[label].most_common()  # Сортировка по частоте
    elif category == 1:
        data = fdist_sw_Obj[label].most_common()
    elif category == 2:
        data = fdist_sw_Adj[label].most_common()
    elif category == 3:
        data = fdist_sw_Noun[label].most_common()
    elif category == 4:
        data = fdist_sw_Participle[label].most_common()

    # Формируем строку вывода с частотностью
    word_list_with_freq = ', '.join([f'{w[0]} ({w[1]})' for w in data])

    return f"Категория: {categories[category]}, Ключевое слово: {label}, Список слов: {word_list_with_freq}"



# Запуск приложения
# Установка макета перед запуском сервера
if __name__ == '__main__':
    app.layout = html.Div([  # Убедимся, что макет установлен перед запуском
        dcc.Graph(
            id='interactive-graph',
            figure=create_figure()
        ),
        html.Div(id='output-text')
    ])

    # Запуск приложения
    app.run_server(debug=True)
