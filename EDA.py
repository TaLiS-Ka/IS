import torchvision
import pandas as pd
import numpy as np

# 1. Загружаем датасет
print("Загружаем EMNIST Letters...")
train_dataset = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=True, download=False
)

# 2. Собираем данные в pandas DataFrame
print("Собираем статистику в DataFrame (это может занять 10-20 секунд)...")
data = []
# Берем подвыборку (первые 10000), перебирая индексы
for i in range(10000):
    img, label = train_dataset[i]
    img_array = np.array(img)

    # Достаем чистое число из тензора
    label_num = int(label.item()) if hasattr(label, 'item') else int(label)

    data.append({
        'label': label_num,
        'label_char': chr(ord('A') + label_num - 1),
        'mean_pixel': float(img_array.mean()),
        'max_pixel': float(img_array.max()),
        'min_pixel': float(img_array.min())
    })

df = pd.DataFrame(data)

# 3. Выводим структуру и пропуски
print("\n--- Структура данных и типы полей ---")
print(df.info())

print("\n--- Пропуски в данных ---")
print(df.isnull().sum())

print("\n--- Базовая статистика ключевых величин (яркость) ---")
print(df[['mean_pixel', 'max_pixel', 'min_pixel']].describe())

# 4. Анализ распределения классов (доли)
print("\n--- Распределение целевых значений (классов) ---")
class_counts = df['label_char'].value_counts().sort_index()
print("Количество картинок по буквам:")
print(class_counts)
print("\nДоли классов (в %):")
print((class_counts / len(df) * 100).round(2))

# 5. Связи между величинами (корреляция)
print("\n--- Корреляция (метка vs средняя яркость) ---")
print(df[['label', 'mean_pixel']].corr())

# 6. Сохраняем машиночитаемую сводку (опционально)
df.to_csv('eda_raw_data.csv', index=False)
print("\nСырые данные для EDA сохранены в eda_raw_data.csv")

# Сохраняем сводку по классам в отдельный файл (это и есть eda_stats.csv)
class_counts.to_csv('eda_stats.csv', header=['count'], index_label='class_name')
print("Сводка по классам сохранена в eda_stats.csv")
