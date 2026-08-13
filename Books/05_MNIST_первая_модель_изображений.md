# 🖼️ Глава 5. MNIST — первая модель изображений

> **Главная цель главы:** перейти от искусственного учебного датасета к настоящим изображениям и обучить первую модель, которая распознаёт рукописные цифры `0–9`.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить, что такое MNIST;
- понимать изображение как Tensor;
- понимать Shape `[1, 28, 28]`;
- пользоваться `Dataset` и `DataLoader`;
- понимать `batch_size` и `shuffle`;
- понимать разницу между Train и Test Dataset;
- превращать изображение `28 × 28` в вектор из `784` чисел;
- понимать, почему на выходе модели 10 logits;
- использовать `CrossEntropyLoss`;
- понимать, почему перед `CrossEntropyLoss` не нужен Softmax;
- получать предсказанный класс через `argmax`;
- обучать модель по Batch;
- считать Accuracy;
- оценивать модель на Test Dataset;
- сохранять и загружать `state_dict()` модели.

# 1. Что такое MNIST

MNIST — классический Dataset рукописных цифр `0–9`.

Каждый пример:

```text
изображение 28 × 28
+
правильная цифра
```

![[Images/chapter05/01_mnist_dataset.png]]

# 2. Изображение как Tensor

После `transforms.ToTensor()` изображение MNIST имеет Shape:

```text
[1, 28, 28]
```

Где:

```text
1  → канал
28 → высота
28 → ширина
```

![[Images/chapter05/02_image_to_tensor.png]]

# 3. Dataset и DataLoader

Dataset хранит примеры:

```text
Dataset[index]
↓
(image, target)
```

DataLoader выдаёт их порциями — Batch.

```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)
```

![[Images/chapter05/03_dataloader_batches.png]]

# 4. Batch

При `batch_size=64` Shape Batch изображений:

```text
[64, 1, 28, 28]
```

А targets:

```text
[64]
```

# 5. Train и Test

```text
Train Dataset
↓
модель учится

Test Dataset
↓
проверяем качество на невиденных данных
```

# 6. Flatten

Обычный Linear-слой ждёт вектор признаков.

```text
28 × 28 = 784
```

Поэтому:

```text
[64, 1, 28, 28]
↓ Flatten
[64, 784]
```

# 7. Архитектура первой MNIST-модели

```text
784
↓
Linear(784 → 128)
↓
ReLU
↓
Linear(128 → 64)
↓
ReLU
↓
Linear(64 → 10)
```

![[Images/chapter05/04_mnist_network.png]]

# 8. Почему 10 выходов

Потому что классов десять:

```text
0 1 2 3 4 5 6 7 8 9
```

Последний слой возвращает 10 logits.

# 9. Logits и argmax

Пример:

```text
[-1.2, 0.3, 2.8, -0.4, ...]
```

Максимальный logit находится у класса `2`.

```python
prediction = logits.argmax(dim=1)
```

![[Images/chapter05/05_multiclass_logits.png]]

# 10. CrossEntropyLoss

Для многоклассовой классификации:

```python
loss_function = nn.CrossEntropyLoss()
```

Правильно:

```python
logits = model(images)
loss = loss_function(logits, targets)
```

Softmax перед `CrossEntropyLoss` вручную не нужен.

# 11. Training Loop по Batch

```python
for images, targets in train_loader:
    logits = model(images)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

![[Images/chapter05/06_mnist_training_pipeline.png]]

# 12. Epoch

Одна Epoch — один полный проход по Train Dataset.

Внутри одной Epoch будет много Batch.

# 13. Accuracy

```python
predictions = logits.argmax(dim=1)
correct = (predictions == targets).sum()
```

# 14. Оценка модели

Для Test Dataset:

```python
model.eval()

with torch.no_grad():
    ...
```

# 15. Overfitting — первое знакомство

Если:

```text
Train Accuracy очень высокая
Test Accuracy заметно ниже
```

модель может переобучаться.

Подробно эту тему разберём позже.

# 16. Сохранение модели

```python
torch.save(
    model.state_dict(),
    "mnist_model.pth",
)
```

# 17. Загрузка модели

```python
model = MNISTNetwork()
model.load_state_dict(
    torch.load("mnist_model.pth")
)
model.eval()
```

# 18. Где хранить Dataset

```text
Almaz_AI/
└── Datasets/
    └── mnist/
```

MNIST скачивается автоматически через `torchvision`.

Сам Dataset в GitHub обычно не добавляют.

# 19. Полный pipeline

```text
MNIST
↓
Dataset
↓
DataLoader
↓
Batch
↓
Flatten
↓
MLP
↓
10 logits
↓
CrossEntropyLoss
↓
Backward
↓
Optimizer
↓
Test Accuracy
```

# 20. 🧪 Что будет в лабораторной работе

Notebook:

```text
Labs/chapter05/05_mnist_classifier.ipynb
```

Мы:

- загрузим MNIST;
- покажем реальные изображения;
- посмотрим Tensor и Shape;
- создадим DataLoader;
- посмотрим один Batch;
- создадим MLP;
- обучим модель;
- построим Loss и Accuracy;
- покажем предсказания;
- найдём ошибки;
- сохраним модель.

# 21. ❓ Самопроверка

1. Что такое MNIST?
2. Какой размер изображения?
3. Что означает `[1,28,28]`?
4. Что делает DataLoader?
5. Что такое Batch?
6. Что делает Flatten?
7. Почему получается 784 признака?
8. Почему последний слой имеет 10 выходов?
9. Что такое logits?
10. Что делает `argmax()`?
11. Для чего нужен `CrossEntropyLoss`?
12. Нужно ли применять Softmax перед ним?
13. Зачем нужен Test Dataset?
14. Что такое Epoch?
15. Как считается Accuracy?
16. Что хранит `state_dict()`?

# 22. 🧩 Практическое задание

Попробуй изменить архитектуру:

```text
784 → 256 → 128 → 10
```

и сравнить её с:

```text
784 → 32 → 10
```

Сравни:

```text
Final Loss
Test Accuracy
число параметров
время обучения
```

# 23. 📌 Что нужно запомнить

```text
Image
↓
Tensor
↓
DataLoader
↓
Batch
↓
Flatten
↓
Model
↓
Logits
↓
CrossEntropyLoss
↓
Backward
↓
Optimizer
```

# 24. 🚀 Следующая глава

# Глава 6. CNN — свёрточные нейронные сети

Там мы перестанем рассматривать изображение как плоский набор из 784 независимых чисел и начнём учитывать его пространственную структуру.
