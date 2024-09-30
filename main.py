import spacy
from termcolor import colored  # библиотека для выделения цветом

nlp = spacy.load("ru_core_news_sm")
f = open('text2.txt', "r", encoding="utf-16")

text = f.read()
text = text.lower()
doc = nlp(text)
word = input('Впишите ключевое слово:')


Sub = []  # глагол подлежащего
Obj = []  # глагол дополнения
Adj = []
Noun = []
word_count = 0  # счетчик появления поискового терма

print([w for w in doc.sents])  # просто для просмотра предложений

# выборка глаголов, существительных и прилагательных
for w in doc.sents:
    print(w)
    for token in w:
        if token.lemma_ == word:
            a = token
            word_count += 1  # увеличиваем счетчик поискового терма
            print(colored(a.text, 'yellow'))  # выделяем поисковый термин цветом
            print(colored(f"{a.text} ({a.pos_}, {a.dep_}, head: {a.head.text})", 'yellow'))  # Для проверки
            if a.dep_ == 'nsubj':  # как объект
                print('sub -', colored(a.head.text, 'yellow'))
                Sub.append(a.head.text)
            if a.dep_ == 'dobj' or a.dep_ == 'obj' or a.dep_ == 'obl':  # как дополнение
                print('obj -', colored(a.head.text, 'yellow'))
                Obj.append(a.head.text)

            print([colored(ch.text, 'yellow') if ch.pos_ in ['ADJ', 'NOUN'] else ch.text for ch in a.children])

            for y in a.children:
                if y.pos_ == 'ADJ':
                    print('adj -', colored(y.text, 'yellow'))  # выделяем прилагательное
                    Adj.append(y.lemma_)

                if y.pos_ == 'NOUN':
                    print('noun -', colored(y.text, 'yellow'))  # выделяем существительное
                    Noun.append(y.text)

# Выводим количество найденных терминов
print(f"Общее количество вхождений ключевого слова '{word}':", word_count)

# Проверка
print('Кол-во глаголов подлежащего', len(Sub), Sub)
print('Кол-во глаголов дополнения', len(Obj), Obj)
print('Кол-во прилагательных', len(Adj), Adj)
print('Кол-во существительных', len(Noun), Noun)


