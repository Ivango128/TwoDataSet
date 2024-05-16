import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('countries of the world.csv')

df = df.dropna()

def get_migration(row):
    row=row.replace(',', '.')
    if float(row) > 0:
        return 1
    else:
        return 0

def get_float(row):
    row=row.replace(',', '.')
    return float(row)

# преводим все данные к численому типу float
df['Net migration'] = df['Net migration'].apply(get_migration)
df['Pop. Density (per sq. mi.)'] = df['Pop. Density (per sq. mi.)'].apply(get_float)
df['Coastline (coast/area ratio)'] = df['Coastline (coast/area ratio)'].apply(get_float)
df['Infant mortality (per 1000 births)'] = df['Infant mortality (per 1000 births)'].apply(get_float)
df['Literacy (%)'] = df['Literacy (%)'].apply(get_float)
df['Phones (per 1000)'] = df['Phones (per 1000)'].apply(get_float)
df['Arable (%)'] = df['Arable (%)'].apply(get_float)
df['Crops (%)'] = df['Crops (%)'].apply(get_float)
df['Other (%)'] = df['Other (%)'].apply(get_float)
df['Climate'] = df['Climate'].apply(get_float)
df['Birthrate'] = df['Birthrate'].apply(get_float)
df['Deathrate'] = df['Deathrate'].apply(get_float)
df['Agriculture'] = df['Agriculture'].apply(get_float)
df['Industry'] = df['Industry'].apply(get_float)
df['Service'] = df['Service'].apply(get_float)

# удаляем столбцы с ненужными данными
df.drop(['Country', 'Region'], axis=1, inplace=True)

print(df.info())
print(df['Net migration'].value_counts())
print(df['Climate'].value_counts())

# Подсчет количества стран с миграцией 0 и 1
zero_count = (df['Net migration'] == 0).sum()
one_count = (df['Net migration'] == 1).sum()

# Создание данных для круговой диаграммы
labels = ['Нет миграции', 'Есть миграция']
sizes = [zero_count, one_count]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)  # "взорвать" первый сегмент

# Создание круговой диаграммы
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Процент стран с разными уровнями миграции')
plt.axis('equal')  # Чтобы диаграмма была круглой
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


X = df.drop('Net migration', axis = 1)
y = df['Net migration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов:', accuracy_score(y_test, y_pred) * 100)

TP, TN, FP, FN = 0, 0, 0, 0

for test, pred in zip(y_test, y_pred):
    if test - pred == 0:
        if test == 1:
            TP += 1
        else:
            TN += 1
    else:
        if test == 1:
            FN += 1
        else:
            FP += 1


#алгоритм распределения по категориям
print('Верный прогноз: есть миграция -', TP, 'нет миграции -', TN)
print('Ошибочный прогноз: есть миграция -', FP, 'нет миграции -', FN)