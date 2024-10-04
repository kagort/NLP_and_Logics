import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


# Загрузка текстового файла и обработка
def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-16') as f:
            text = f.read()
        return text
    except FileNotFoundError:
        print("Файл не найден. Проверьте путь и попробуйте снова.")
        return None
    except UnicodeDecodeError:
        print("Не удалось прочитать файл. Проверьте кодировку.")
        return None


# Функция для подсчета ключевых слов в каждом предложении
def count_keywords_in_sentences(text, keywords):
    sentences = sent_tokenize(text)
    data = {kw: [] for kw in keywords}
    data['Предложение'] = []

    for sentence in sentences:
        word_counts = {kw: word_tokenize(sentence.lower()).count(kw.lower()) for kw in keywords}
        for kw in keywords:
            data[kw].append(word_counts[kw])
        data['Предложение'].append(sentence)

    df = pd.DataFrame(data)
    return df.set_index('Предложение')


# Ввод и обработка ключевых слов
file_path = input("Введите путь к файлу с текстом: ")
keywords = input("Введите ключевые слова через запятую: ").split(',')
keywords = [kw.strip() for kw in keywords]

# Загрузка текста и подсчет ключевых слов
text = load_text(file_path)
if text:
    df = count_keywords_in_sentences(text, keywords)

    # Фильтрация предложений, содержащих хотя бы одно ключевое слово
    df_filtered = df[(df > 0).any(axis=1)]

    # Рассчет коэффициентов плотности
    density = df_filtered.sum() / len(df_filtered)  # Относительная частота
    density = density.round(2)  # Округление до 2 знаков после запятой

    # Построение тепловой карты
    plt.figure(figsize=(12, 8))
    heatmap = sns.heatmap(df_filtered, annot=True, cmap="YlGnBu", cbar=True)
    plt.title("Тепловая карта распределения ключевых слов по предложениям")
    plt.xlabel("Ключевые слова")
    plt.ylabel("Предложения")
    plt.yticks(rotation=0)  # Поворот меток оси Y на 0 градусов

    # Добавление коэффициентов плотности
    for i, kw in enumerate(density.index):
        heatmap.text(i + 0.5, -0.5, f'Плотность: {density[kw]}',
                     ha='center', va='center', fontsize=10, color='black')

    plt.tight_layout()
    plt.show()
