# Задание 1
# Импортируйте библиотеку Pandas и дайте ей псевдоним pd. Создайте датафрейм authors со столбцами
# author_id и author_name, в которых соответственно содержатся данные: [1, 2, 3] и ['Тургенев',
# 'Чехов', 'Островский'].
# Затем создайте датафрейм book cо столбцами author_id, book_title и price, в которых соответственно
# содержатся данные:
# [1, 1, 1, 2, 2, 3, 3],
# ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза',
# 'Таланты и поклонники']

import numpy as np
import pandas as pd

authors = pd.DataFrame({'author_id': [1, 2, 3],
                        'author_name': ['Тургенев', 'Чехов', 'Островский']}, columns=['author_id', 'author_name'])
book = pd.DataFrame({'author_id': [1, 1, 1, 2, 2, 3, 3],
                     'book_title': ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой',
                                    'Гроза', 'Таланты и поклонники'],
                     'price': [40, 50, 60, 70, 80, 90, 100]}, columns=['author_id', 'book_title', 'price'])

# Задание 2
# Получите датафрейм authors_price, соединив датафреймы authors и books по полю author_id.

authors_price = pd.merge(authors, book, on='author_id', how='left')

# Задание 3
# Создайте датафрейм top5, в котором содержатся строки из authors_price с пятью самыми дорогими книгами.

top5 = pd.DataFrame(authors_price.tail())
print(top5)

# Задание 4
# Создайте датафрейм authors_stat на основе информации из authors_price. В датафрейме authors_stat
# должны быть четыре столбца:
# author_name, min_price, max_price и mean_price,
# в которых должны содержаться соответственно имя автора, минимальная, максимальная и средняя цена
# на книги этого автора.

authors_stat = pd.DataFrame(authors_price.groupby(['author_name']).agg({'price': ['min', 'max', 'mean']}))
print(authors_stat.rename(columns={'min': 'min_price', 'max': 'max_price', 'mean': 'mean_price'}))

# Задание 5**
# Создайте новый столбец в датафрейме authors_price под названием cover, в нем будут располагаться
# данные о том, какая обложка у данной книги - твердая или мягкая. В этот столбец поместите данные из
# следующего списка:
# ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая'].
# Просмотрите документацию по функции pd.pivot table с помощью вопросительного знака.
# Для каждого автора посчитайте суммарную стоимость книг в твердой и мягкой обложке. Используйте для
# этого функцию pd.pivot_table. При этом столбцы должны называться "твердая" и "мягкая", а индексами
# должны быть фамилии авторов. Пропущенные значения стоимостей заполните нулями, при необходимости
# загрузите библиотеку Numpy.
# Назовите полученный датасет book_info и сохраните его в формат pickle под названием "book_info.pkl".
# Затем загрузите из этого файла датафрейм и назовите его book_info2. Удостоверьтесь, что датафреймы
# book_info и book_info2 идентичны.


authors_price['cover'] = ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
book_info = pd.DataFrame(
    pd.pivot_table(authors_price, values='price', index='author_name', columns='cover', aggfunc=np.sum, fill_value=0))
print(book_info)
book_info.to_pickle('book_info.pkl')
book_info2 = pd.read_pickle('book_info.pkl')
book_info.equals(book_info2)
