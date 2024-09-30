import spacy
from colorama import Fore, Style, init
from nltk.corpus import stopwords  # Импортируем стоп-слова
from nltk.probability import FreqDist  # Импортируем FreqDist для частотного анализа
import nltk

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

                # Выделение поискового термина желтым
                highlighted_term = Fore.YELLOW + a.text + Style.RESET_ALL
                print(highlighted_term)
                print(Fore.YELLOW + f"{a.text} ({a.pos_}, {a.dep_}, head: {a.head.text})" + Style.RESET_ALL)

                if a.dep_ == 'nsubj':  # как подлежащее
                    sub_head = Fore.GREEN + a.head.text + Style.RESET_ALL  # Глаголы подлежащего - зеленый
                    print('sub -', sub_head)
                    Sub[word].append(a.head.text)  # Записываем только для текущего ключевого слова
                if a.dep_ in ('dobj', 'obj', 'obl'):  # как дополнение
                    obj_head = Fore.CYAN + a.head.text + Style.RESET_ALL  # Глаголы дополнения - голубой
                    print('obj -', obj_head)
                    Obj[word].append(a.head.text)  # Записываем только для текущего ключевого слова

                # Вывод зависимых слов по отдельности с выделением прилагательных, существительных и причастий
                for ch in a.children:
                    if ch.pos_ == 'ADJ':  # Прилагательные
                        highlighted_adj = Fore.MAGENTA + ch.text + Style.RESET_ALL
                        print('adj -', highlighted_adj)
                        Adj[word].append(ch.lemma_)
                    elif ch.pos_ == 'NOUN':  # Существительные
                        highlighted_noun = Fore.RED + ch.text + Style.RESET_ALL
                        print('noun -', highlighted_noun)
                        Noun[word].append(ch.text)
                    elif ch.pos_ == 'VERB':  # Причастия
                        highlighted_participle = Fore.BLUE + ch.text + Style.RESET_ALL
                        print('participle -', highlighted_participle)
                        Participle[word].append(ch.text)
                    else:
                        print(ch.text)

# Вывод общего количества вхождений поисковых терминов
for word, count in word_counts.items():
    print(f"\nОбщее количество вхождений ключевого слова '{word}': {count}")

# Функция для поэлементного вывода цветных списков
def print_colored_list(title, word, words, color):
    print(f"{title} для термина '{word}': {len(words)}")
    for word in words:
        print(color + word + Style.RESET_ALL)

# Вывод итогов для каждого ключевого слова
for word in words:
    print_colored_list('Глаголы подлежащего', word, Sub[word], Fore.GREEN)
    print_colored_list('Глаголы дополнения', word, Obj[word], Fore.CYAN)
    print_colored_list('Прилагательные', word, Adj[word], Fore.MAGENTA)
    print_colored_list('Существительные', word, Noun[word], Fore.RED)
    print_colored_list('Причастия', word, Participle[word], Fore.BLUE)

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

# Визуализация
fdist_sw_Sub = {word: FreqDist(filt_Sub[word]) for word in words}
fdist_sw_Obj = {word: FreqDist(filt_Obj[word]) for word in words}
fdist_sw_Adj = {word: FreqDist(filt_Adj[word]) for word in words}
fdist_sw_Noun = {word: FreqDist(filt_Noun[word]) for word in words}
fdist_sw_Participle = {word: FreqDist(filt_Participle[word]) for word in words}

# Вывод частотных слов для каждого ключевого слова
for word in words:
    print(f"\nЧастотные слова для термина '{word}':")
    print(f"Глаголы подлежащего: {fdist_sw_Sub[word].most_common(60)}")
    print(f"Глаголы дополнения: {fdist_sw_Obj[word].most_common(60)}")
    print(f"Прилагательные: {fdist_sw_Adj[word].most_common(60)}")
    print(f"Существительные: {fdist_sw_Noun[word].most_common(60)}")
    print(f"Причастия: {fdist_sw_Participle[word].most_common(60)}")
