import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

# Отключаем предупреждения
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sns.set_theme(style="whitegrid")

# 1. Загрузка данных
print("1. Загрузка данных...")
train_ds = torchvision.datasets.EMNIST(root='./data', split='letters', train=True, download=False)
test_ds = torchvision.datasets.EMNIST(root='./data', split='letters', train=False, download=False)

# Исправление: берем случайные индексы из ВСЕГО датасета, чтобы модель увидела все буквы
np.random.seed(42)
train_indices = np.random.choice(len(train_ds), 10000, replace=False)
test_indices = np.random.choice(len(test_ds), 3000, replace=False)

X_train = np.array([np.array(train_ds[i][0]).flatten() for i in train_indices])
y_train = np.array([train_ds[i][1] for i in train_indices])

X_test = np.array([np.array(test_ds[i][0]).flatten() for i in test_indices])
y_test = np.array([test_ds[i][1] for i in test_indices])

class_names = [chr(ord('A') + i - 1) for i in range(1, 27)]

# 2. ГРАФИК 1: Дисбаланс классов
print("2. Строим график дисбаланса...")
class_counts = Counter(y_train)
counts = [class_counts.get(i, 0) for i in range(1, 27)]

plt.figure(figsize=(14, 6))
ax = sns.barplot(x=class_names, y=counts, hue=class_names, palette="viridis", legend=False)
plt.title('Проблема варианта: Дисбаланс классов в EMNIST Letters (Train)', fontsize=15)
plt.xlabel('Буква (класс)', fontsize=12)
plt.ylabel('Количество изображений', fontsize=12)

for i, v in enumerate(counts):
    ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plot_imbalance.png', dpi=300)
plt.show()

# 3. Обучение модели (LogReg на PCA)
print("3. Обучение LogReg на PCA(64)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=64)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_pca, y_train)
y_pred = clf.predict(X_test_pca)

# 4. ГРАФИК 2: Путаница классов (Confusion Matrix)
print("4. Строим матрицу ошибок...")
cm = confusion_matrix(y_test, y_pred, labels=range(1, 27))

cm_percent = np.zeros_like(cm, dtype=float)
row_sums = cm.sum(axis=1)
for i in range(len(cm)):
    if row_sums[i] > 0:
        cm_percent[i, :] = cm[i, :] / row_sums[i] * 100

plt.figure(figsize=(14, 12))
sns.heatmap(cm_percent, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Процент ошибок (%)'})
plt.title('Проблема варианта: Путаница классов (LogReg на PCA, %)', fontsize=16)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.ylabel('Истинный класс', fontsize=12)
plt.tight_layout()
plt.savefig('plot_confusion.png', dpi=300)
plt.show()

print("Готово! Графики сохранены: plot_imbalance.png и plot_confusion.png")
