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
word = input('Впишите ключевое слово:').lower()

Sub = []  # глагол подлежащего
Obj = []  # глагол дополнения
Adj = []
Noun = []
Participle = []  # Добавим список для причастий
word_count = 0  # Счетчик вхождений поискового термина

# Нумерация предложений
print("Список предложений в тексте:")
for i, sent in enumerate(doc.sents, 1):
    print(f"{i}. {sent}")

print("\nНачало анализа...\n")

# Выборка глаголов, существительных, прилагательных и причастий
for sent in doc.sents:
    print(sent)
    for token in sent:
        if token.lemma_ == word:
            a = token
            word_count += 1  # Увеличиваем счетчик поискового термина

            # Выделение поискового термина желтым
            highlighted_term = Fore.YELLOW + a.text + Style.RESET_ALL
            print(highlighted_term)
            print(Fore.YELLOW + f"{a.text} ({a.pos_}, {a.dep_}, head: {a.head.text})" + Style.RESET_ALL)

            if a.dep_ == 'nsubj':  # как подлежащее
                sub_head = Fore.GREEN + a.head.text + Style.RESET_ALL  # Глаголы подлежащего - зеленый
                print('sub -', sub_head)
                Sub.append(a.head.text)
            if a.dep_ in ('dobj', 'obj', 'obl'):  # как дополнение
                obj_head = Fore.CYAN + a.head.text + Style.RESET_ALL  # Глаголы дополнения - голубой
                print('obj -', obj_head)
                Obj.append(a.head.text)

            # Вывод зависимых слов по отдельности с выделением прилагательных, существительных и причастий
            for ch in a.children:
                if ch.pos_ == 'ADJ':
                    highlighted_adj = Fore.MAGENTA + ch.text + Style.RESET_ALL  # Прилагательные - фиолетовый
                    print('adj -', highlighted_adj)
                    Adj.append(ch.lemma_)
                elif ch.pos_ == 'NOUN':
                    highlighted_noun = Fore.RED + ch.text + Style.RESET_ALL  # Существительные - красный
                    print('noun -', highlighted_noun)
                    Noun.append(ch.text)
                elif ch.pos_ == 'VERB':  # Учтем причастия, которые помечены как глаголы
                    highlighted_participle = Fore.BLUE + ch.text + Style.RESET_ALL  # Причастия - синие
                    print('participle -', highlighted_participle)
                    Participle.append(ch.text)
                else:
                    print(ch.text)

# Вывод общего количества вхождений поискового термина
print(f"\nОбщее количество вхождений ключевого слова '{word}': {word_count}\n")

# Функция для поэлементного вывода цветных списков
def print_colored_list(title, words, color):
    print(f"{title}: {len(words)}")
    for word in words:
        print(color + word + Style.RESET_ALL)

# Проверка с выводом итоговых списков с цветами
print_colored_list('Глаголы подлежащего', Sub, Fore.GREEN)
print_colored_list('Глаголы дополнения', Obj, Fore.CYAN)
print_colored_list('Прилагательные', Adj, Fore.MAGENTA)
print_colored_list('Существительные', Noun, Fore.RED)
print_colored_list('Причастия', Participle, Fore.BLUE)  # Выводим список причастий

# Лемматизация списков Sub и Obj
lemmatized_Sub = [" ".join([token.lemma_ for token in nlp(text)]) for text in Sub]
lemmatized_Obj = [" ".join([token.lemma_ for token in nlp(text)]) for text in Obj]

# Работа со стоп-словами
stop_w = stopwords.words("russian")
stop_w.extend(['�', '-', '―', '”', 'запах', 'вкус'])

filt_Sub = [w for w in lemmatized_Sub if w not in stop_w]
filt_Obj = [w for w in lemmatized_Obj if w not in stop_w]
filt_Adj = [w for w in Adj if w not in stop_w]
filt_Noun = [w for w in Noun if w not in stop_w]
filt_Participle = [w for w in Participle if w not in stop_w]  # Фильтрация причастий

# Удаление повторов и сортировка
sub = sorted(list(set(filt_Sub)))
obj = sorted(list(set(filt_Obj)))
both = sub + obj

adj = sorted(list(set(filt_Adj)))
noun = sorted(list(set(filt_Noun)))
participle = sorted(list(set(filt_Participle)))

# Визуализация
fdist_sw_Sub = FreqDist(filt_Sub)
fdist_sw_Obj = FreqDist(filt_Obj)
fdist_sw_Adj = FreqDist(filt_Adj)
fdist_sw_Noun = FreqDist(filt_Noun)
fdist_sw_Participle = FreqDist(filt_Participle)  # Добавим частотный анализ причастий

print("\nЧастотные слова:")
print(f"Глаголы подлежащего: {fdist_sw_Sub.most_common(60)}")
print(f"Глаголы дополнения: {fdist_sw_Obj.most_common(60)}")
print(f"Прилагательные: {fdist_sw_Adj.most_common(60)}")
print(f"Существительные: {fdist_sw_Noun.most_common(60)}")
print(f"Причастия: {fdist_sw_Participle.most_common(60)}")
