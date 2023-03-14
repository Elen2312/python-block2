# Задание 1
# Импортируйте библиотеки pandas и numpy.
# Загрузите "Boston House Prices dataset" из встроенных наборов данных библиотеки sklearn.
# Разбейте эти датафреймы на тренировочные (X_train, y_train) и тестовые (X_test, y_test) с помощью функции
# train_test_split так, чтобы размер тестовой выборки составлял 30% от всех данных, при этом аргумент random state
# должен быть равен 42.
# Создайте модель линейной регрессии под названием lr с помощью класса LinearRegression из модуля sklearn.linear_model.
# Обучите модель на тренировочных данных (используйте все признаки) и сделайте предсказание на тестовых.

import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

boston = load_boston()
boston.keys()
data = boston.data
target = boston.target
feature_names = boston.feature_names
x = pd.DataFrame(data, columns=feature_names)
y = pd.DataFrame(target, columns=['price'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
check_test = pd.DataFrame({'y_test': y_test['price'],
                           'y_pred': y_pred.flatten()})
print(check_test.head(10))
initial_mse = mean_squared_error(y_test, y_pred)
print(f'Средняя квадратичная ошибка - {initial_mse}')
print(f'Средняя абсолютная ошибка - {mean_absolute_error(y_test, y_pred)}')
print(f'R2 - {r2_score(y_test, y_pred)}')

# plt.barh(x_train.columns, lr.coef_.flatten())
# plt.xlabel('Вес признака')
# plt.ylabel('Признак')
# plt.show()
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = scaler.transform(x_test)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)
lr.fit(x_train_scaled, y_train)
# plt.barh(x_train.columns, lr.coef_.flatten())
# plt.xlabel('Вес признака')
# plt.ylabel('Признак')
# plt.show()
feats = ['LSTAT', 'B', 'PTRATIO', 'TAX', 'RAD', 'DIS', 'RM', 'NOX', 'CHAS', 'ZN', 'CRIM']


def create_model(x_train, y_train, x_test, y_test, feats, model):
    model.fit(x_train.loc[:, feats], y_train)
    y_pred = model.predict(x_test.loc[:, feats])
    mse = mean_squared_error(y_test, y_pred)
    return mse


print(f'Средняя квадратичная ошибка без признаков меньше 0,5 - {create_model(x_train_scaled, y_train, x_test_scaled, y_test, feats, LinearRegression())}')

model = Lasso(alpha=0.003)
print(f'Линейная регрессия с L1-регуляризацией - {create_model(x_train_scaled, y_train, x_test_scaled, y_test, feats, model)}')
model = Ridge(alpha=0.001)
print(f'Линейная регрессия с L2-регуляризацией - {create_model(x_train_scaled, y_train, x_test_scaled, y_test, feats, model)}')

# Задание 2
# Создайте модель под названием model с помощью класса RandomForestRegressor из модуля sklearn.ensemble.
# Сделайте агрумент n_estimators равным 1000,
# max_depth должен быть равен 12 и random_state сделайте равным 42.
# Обучите модель на тренировочных данных аналогично тому, как вы обучали модель LinearRegression,
# но при этом в метод fit вместо датафрейма y_train поставьте y_train.values[:, 0],
# чтобы получить из датафрейма одномерный массив Numpy,
# так как для класса RandomForestRegressor в данном методе для аргумента y предпочтительно применение массивов вместо
# датафрейма.
# Сделайте предсказание на тестовых данных и посчитайте R2. Сравните с результатом из предыдущего задания.
# Напишите в комментариях к коду, какая модель в данном случае работает лучше.


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=1000, max_depth=12, random_state=42)
model.fit(x_train, y_train.values[:, 0])
y_pred_model = model.predict(x_test)
check_test_model = pd.DataFrame({'y_test': y_test['price'],
                                 'y_pred_model': y_pred_model.flatten()})
print(check_test_model.head(10))
mse_model = mean_squared_error(y_test, y_pred_model)
print(f'Средняя квадратичная ошибка - {mse_model}')
print(f'R2 - {r2_score(y_test, y_pred_model)}')
# Данная модель лучше, так как ее значение меньше (чем ближе к нулю, тем модель лучше)

# *Задание 3
# Вызовите документацию для класса RandomForestRegressor,
# найдите информацию об атрибуте feature_importances_.
# С помощью этого атрибута найдите сумму всех показателей важности,
# установите, какие два признака показывают наибольшую важность.

# help(RandomForestRegressor)
print(model.feature_importances_)
fi = pd.DataFrame({'name': x.columns, 'feature_importance': model.feature_importances_})
print(fi)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(fi['name'].value_counts().index, model.feature_importances_)
ax.set_title("Важность признаков")
plt.xlabel('Признак')
ax.set_ylabel('Важность')
# plt.show()

print(fi.nlargest(2, 'feature_importance'))

# *Задание 4
# В этом задании мы будем работать с датасетом, с которым мы уже знакомы по домашнему заданию по библиотеке Matplotlib,
# это датасет Credit Card Fraud Detection. Для этого датасета мы будем решать задачу классификации - будем определять,
# какие из транзакции по кредитной карте являются мошенническими. Данный датасет сильно несбалансирован (так как случаи
# мошенничества относительно редки), так что применение метрики accuracy не принесет пользы и не поможет выбрать лучшую
# модель. Мы будем вычислять AUC, то есть площадь под кривой ROC.
# Импортируйте из соответствующих модулей RandomForestClassifier, GridSearchCV и train_test_split.
# Загрузите датасет creditcard.csv и создайте датафрейм df.
# С помощью метода value_counts с аргументом normalize=True убедитесь в том, что выборка несбалансированна. Используя
# метод info, проверьте, все ли столбцы содержат числовые данные и нет ли в них пропусков. Примените следующую
# настройку, чтобы можно было просматривать все столбцы датафрейма:
# pd.options.display.max_columns = 100.
# Просмотрите первые 10 строк датафрейма df.
# Создайте датафрейм X из датафрейма df, исключив столбец Class.
# Создайте объект Series под названием y из столбца Class.
# Разбейте X и y на тренировочный и тестовый наборы данных при помощи функции train_test_split, используя аргументы:
# test_size=0.3, random_state=100, stratify=y.
# У вас должны получиться объекты X_train, X_test, y_train и y_test.
# Просмотрите информацию об их форме.
# Для поиска по сетке параметров задайте такие параметры:
# parameters = [{'n_estimators': [10, 15],
# 'max_features': np.arange(3, 5),
# 'max_depth': np.arange(4, 7)}]
# Создайте модель GridSearchCV со следующими аргументами:
# estimator=RandomForestClassifier(random_state=100),
# param_grid=parameters,
# scoring='roc_auc',
# cv=3.
# Обучите модель на тренировочном наборе данных (может занять несколько минут).
# Просмотрите параметры лучшей модели с помощью атрибута best_params_.
# Предскажите вероятности классов с помощью полученной модели и метода predict_proba.
# Из полученного результата (массив Numpy) выберите столбец с индексом 1 (вероятность класса 1) и запишите в массив
# y_pred_proba. Из модуля sklearn.metrics импортируйте метрику roc_auc_score.
# Вычислите AUC на тестовых данных и сравните с результатом, полученным на тренировочных данных, используя в качестве
# аргументов массивы y_test и y_pred_proba.

# *Дополнительные задания:
# 1). Загрузите датасет Wine из встроенных датасетов sklearn.datasets с помощью функции load_wine в переменную data.

from sklearn.datasets import load_wine

data = load_wine()

# 2). Полученный датасет не является датафреймом. Это структура данных, имеющая ключи аналогично словарю. Просмотрите
# тип данных этой структуры данных и создайте список data_keys, содержащий ее ключи.

print(type(data))
data_keys = data.keys()
print(data_keys)

# 3). Просмотрите данные, описание и названия признаков в датасете. Описание нужно вывести в виде привычного,
# аккуратно оформленного текста, без обозначений переноса строки, но с самими переносами и т.д.

print(data.DESCR)

# 4). Сколько классов содержит целевая переменная датасета? Выведите названия классов.

print(data.target_names)
print(len(set(data.target)))

# 5). На основе данных датасета (они содержатся в двумерном массиве Numpy) и названий признаков создайте датафрейм
# под названием X.

x = pd.DataFrame(data.data, columns=data.feature_names)
print(x.head())

# 6). Выясните размер датафрейма X и установите, имеются ли в нем пропущенные значения.

x.info()

# 7). Добавьте в датафрейм поле с классами вин в виде чисел, имеющих тип данных numpy.int64. Название поля - 'target'.

x['target'] = data['target'].astype(np.int64)
x.info()

# 8). Постройте матрицу корреляций для всех полей X. Дайте полученному датафрейму название X_corr.

X_corr = x.corr()
print(X_corr)

# 9). Создайте список high_corr из признаков, корреляция которых с полем target по абсолютному значению превышает 0.5
# (причем, само поле target не должно входить в этот список).

high_corr = X_corr['target']
high_corr = high_corr[np.abs(high_corr) > 0.5].drop('target', axis=0)
high_corr = list(high_corr.index)
print(high_corr)

# 10). Удалите из датафрейма X поле с целевой переменной. Для всех признаков, названия которых содержатся в списке
# high_corr, вычислите квадрат их значений и добавьте в датафрейм X соответствующие поля с суффиксом '_2',
# добавленного к первоначальному названию признака. Итоговый датафрейм должен содержать все поля, которые, были в нем
# изначально, а также поля с признаками из списка high_corr, возведенными в квадрат. Выведите описание полей
# датафрейма X с помощью метода describe.

x = x.drop('target', axis=1)
for i in high_corr:
    x[i+'_2'] = x[i]**2
print(x.describe())
