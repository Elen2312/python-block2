# Задание 1
# Импортируйте библиотеки pandas, numpy и matplotlib.
# Загрузите "Boston House Prices dataset" из встроенных наборов
# данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test)
# с помощью функции train_test_split так, чтобы размер тестовой выборки
# составлял 20% от всех данных, при этом аргумент random_state должен быть равен 42.
# Масштабируйте данные с помощью StandardScaler.
# Постройте модель TSNE на тренировочный данных с параметрами:
# n_components=2, learning_rate=250, random_state=42.
# Постройте диаграмму рассеяния на этих данных.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

boston = load_boston()
x = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

tsne = TSNE(n_components=2, learning_rate=250, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_scaled)
print('До:\t{}'.format(X_train_scaled.shape))
print('После:\t{}'.format(X_train_tsne.shape))

plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1])
# plt.show()

# # Задание 2
# # С помощью KMeans разбейте данные из тренировочного набора на 3 кластера,
# # используйте все признаки из датафрейма X_train.
# # Параметр max_iter должен быть равен 100, random_state сделайте равным 42.
# # Постройте еще раз диаграмму рассеяния на данных, полученных с помощью TSNE,
# # и раскрасьте точки из разных кластеров разными цветами.
# # Вычислите средние значения price и CRIM в разных кластерах.

kmeans = KMeans(n_clusters=3, max_iter=100, random_state=42)
labels_train = kmeans.fit_predict(X_train_scaled)
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=labels_train)
# plt.show()
print(f'Средняя цена - {y_train.mean()}')
print(f'Средняя цена 1 кластера - {y_train[labels_train == 0].mean()}')
print(f'Средняя цена 2 кластера - {y_train[labels_train == 1].mean()}')
print(f'Средняя цена 3 кластера - {y_train[labels_train == 2].mean()}')
print('Кластер 1: {}'.format(X_train.loc[labels_train == 0, 'CRIM'].mean()))
print('Кластер 2: {}'.format(X_train.loc[labels_train == 1, 'CRIM'].mean()))
print('Кластер 3: {}'.format(X_train.loc[labels_train == 2, 'CRIM'].mean()))

# # *Задание 3
# # Примените модель KMeans, построенную в предыдущем задании,
# # к данным из тестового набора.
# # Вычислите средние значения price и CRIM в разных кластерах на тестовых данных.

X_test_tsne = tsne.fit_transform(X_test_scaled)
labels_test = kmeans.fit_predict(X_test_scaled)
plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=labels_test)
# plt.show()
print(f'ТЕСТОВЫЕ ДАННЫЕ: \nСредняя цена - {y_test.mean()}')
print(f'Средняя цена 1 кластера - {y_test[labels_test == 0].mean()}')
print(f'Средняя цена 2 кластера - {y_test[labels_test == 1].mean()}')
print(f'Средняя цена 3 кластера - {y_test[labels_test == 2].mean()}')
print('Кластер 1: {}'.format(X_test.loc[labels_test == 0, 'CRIM'].mean()))
print('Кластер 2: {}'.format(X_test.loc[labels_test == 1, 'CRIM'].mean()))
print('Кластер 3: {}'.format(X_test.loc[labels_test == 2, 'CRIM'].mean()))

# # *Задание 4
# # Обучите любую модель регрессии на этом же датасете. Добавьте новый признак - метка кластера, которую вы уже получили
# # применив модель кластеризации к этим данным. Сравнить качество без метки кластера и с ней по отложенной выборке.
#
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
# Не поняла условие про добавление нового признака - метка кластера

# *Задание 5
# Загрузите "wine dataset" из встроенных наборов данных библиотеки sklearn.
# Создайте датафреймы X и y из этих данных.
# Разбейте эти датафреймы на тренировочные и тестовые.
# Масштабируйте данные.
# Постройте модель понижения размерности на тренировочный данных, визуализируйте с помощью диаграммы рассеяния,
# подберите оптимальные гиперпараметры, чтобы сегментов на графике было столько, сколько классов в данных.
# С помощью модели кластеризации (поэкспериментируйте и с другими моделями, не только с KMeans) разбейте данные из
# тренировочного набора на необходимое количество кластеров.
# Постройте диаграмму рассеяния на данных, полученных с помощью понижения размерности и раскрасьте точки из разных
# кластеров разными цветами.
# Убедитесь с помощью визуализации, что разбиение данных из тестового набора получилось успешным.
# Сделайте вывод, какие модели лучше себя показывают на этой задаче.

# *Задание 6
# Используйте "Olivetti faces data-set from AT&T" из встроенных наборов данных библиотеки sklearn.
# Обучить любую модель классификации на этом датасете до применения PCA (количество компонент подберите самостоятельно)
# и после него. Сравнить качество классификации по отложенной выборке.
