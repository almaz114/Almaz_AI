---
title: "Глава 6. CNN — свёрточные нейронные сети"
project: "Almaz_AI"
chapter: 6
tags: [cnn, convolution, pooling, pytorch, images, mnist, feature_maps]
---

# 🧩 Глава 6. CNN — свёрточные нейронные сети

> **Главная цель главы:** понять, почему обычная полносвязная сеть не учитывает пространственную структуру изображения, и научиться использовать свёрточные слои для выделения локальных признаков — линий, контуров, углов и более сложных форм.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить, зачем нужны CNN;
- понимать отличие MLP от CNN;
- понимать, что такое свёртка;
- понимать роль Kernel / Filter;
- понимать `stride`;
- понимать `padding`;
- понимать, что такое Feature Map;
- понимать `Conv2d`;
- понимать `MaxPool2d`;
- отслеживать Shape после свёрточных слоёв;
- собирать простую CNN в PyTorch;
- обучать CNN на MNIST;
- визуализировать Feature Maps.

# 1. Почему после MLP нужен CNN

В главе 5 мы делали:

```text
Image [1, 28, 28]
↓
Flatten
↓
784 числа
↓
Linear
```

После `Flatten` модель теряет явную информацию о пространственном расположении пикселей.

CNN работает с локальной структурой изображения.

![[Images/chapter06/01_mlp_vs_cnn.png]]

# 2. Kernel / Filter

Kernel — маленькая матрица обучаемых весов.

Например:

```text
3 × 3
```

Фильтр перемещается по изображению и реагирует на локальные структуры.

![[Images/chapter06/02_convolution_kernel.png]]

# 3. Что такое свёртка

```text
Patch изображения
×
Kernel
↓
сумма произведений
↓
одно новое число
```

Повторяя это по всему изображению, получаем `Feature Map`.

# 4. Conv2d

Пример:

```python
nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
)
```

Это значит:

```text
1 входной канал
↓
8 обучаемых фильтров
↓
8 Feature Maps
```

# 5. Shape входа CNN

Для Batch MNIST:

```text
[64, 1, 28, 28]
```

Порядок PyTorch:

```text
[N, C, H, W]
```

# 6. Stride

`stride` — размер шага Kernel.

```text
stride=1 → шаг 1 пиксель
stride=2 → шаг 2 пикселя
```

# 7. Padding

Без Padding при Kernel 3×3:

```text
28 × 28
↓
26 × 26
```

С:

```python
padding=1
```

размер можно сохранить:

```text
28 × 28
```

![[Images/chapter06/03_stride_padding_featuremap.png]]

# 8. ReLU после Conv2d

Частый блок:

```text
Conv2d
↓
ReLU
```

ReLU добавляет нелинейность, как и в предыдущих главах.

# 9. Max Pooling

```python
nn.MaxPool2d(2)
```

Для области:

```text
1 4
2 3
```

результат:

```text
4
```

Pooling уменьшает пространственный размер.

![[Images/chapter06/04_max_pooling.png]]

# 10. Первый CNN-блок

```python
nn.Conv2d(1, 8, kernel_size=3, padding=1)
nn.ReLU()
nn.MaxPool2d(2)
```

Shapes:

```text
[64,1,28,28]
↓
[64,8,28,28]
↓
[64,8,14,14]
```

# 11. Второй CNN-блок

```python
nn.Conv2d(8, 16, kernel_size=3, padding=1)
nn.ReLU()
nn.MaxPool2d(2)
```

Shapes:

```text
[64,8,14,14]
↓
[64,16,14,14]
↓
[64,16,7,7]
```

# 12. Flatten после CNN

После второго Pooling:

```text
[64,16,7,7]
```

Количество признаков на пример:

```text
16 × 7 × 7 = 784
```

После `Flatten`:

```text
[64,784]
```

# 13. Полная CNN

```text
Input [1,28,28]
↓
Conv2d(1→8)
↓
ReLU
↓
MaxPool
↓
Conv2d(8→16)
↓
ReLU
↓
MaxPool
↓
Flatten
↓
Linear(784→64)
↓
ReLU
↓
Linear(64→10)
```

![[Images/chapter06/05_cnn_architecture.png]]

# 14. Почему CNN подходит для изображений

CNN использует:

```text
локальность
пространственную структуру
повторное использование одного фильтра
```

Один и тот же Kernel применяется в разных местах изображения.

Это называется:

```text
weight sharing
```

# 15. Feature Maps

Разные Filters учатся реагировать на разные признаки.

Очень упрощённо:

```text
ранние слои → линии и края
средние слои → углы и части формы
глубокие слои → сложные комбинации
```

![[Images/chapter06/06_feature_maps.png]]

# 16. CNN обучается тем же способом

```text
Forward
↓
Loss
↓
zero_grad()
↓
backward()
↓
optimizer.step()
```

Так как задача остаётся многоклассовой, используем:

```python
nn.CrossEntropyLoss()
```

# 17. Shapes всей модели

```text
Input:
[64,1,28,28]

Conv1:
[64,8,28,28]

Pool1:
[64,8,14,14]

Conv2:
[64,16,14,14]

Pool2:
[64,16,7,7]

Flatten:
[64,784]

Output:
[64,10]
```

# 18. 🧪 Что будет в Notebook

```text
Labs/chapter06/06_cnn_mnist.ipynb
```

Мы:

- загрузим MNIST;
- создадим `Conv2d`;
- посмотрим Shapes;
- разберём Pooling;
- создадим CNN;
- обучим её;
- построим Loss и Accuracy;
- визуализируем Feature Maps;
- сохраним модель.

# 19. ❓ Самопроверка

1. Почему MLP не идеально подходит для изображений?
2. Что такое Kernel?
3. Что такое Feature Map?
4. Что означает `in_channels`?
5. Что означает `out_channels`?
6. Что делает `kernel_size`?
7. Что такое `stride`?
8. Что такое `padding`?
9. Что делает `MaxPool2d`?
10. Что означает `[N,C,H,W]`?
11. Что такое Weight Sharing?
12. Почему после CNN нужен Flatten перед Linear?
13. Почему `CrossEntropyLoss` остаётся той же?
14. Что можно увидеть на Feature Maps?

# 20. 🧩 Практическое задание

Попробуй заменить:

```text
1 → 8 → 16
```

на:

```text
1 → 16 → 32
```

и сравнить:

```text
число параметров
время обучения
Test Accuracy
```

# 21. 📌 Что нужно запомнить

```text
Image
↓
Conv2d
↓
Feature Maps
↓
ReLU
↓
Pooling
↓
Conv2d
↓
Pooling
↓
Flatten
↓
Linear
↓
Logits
```

# 22. 🚀 Следующая глава

# Глава 7. Текст, токены и Embeddings

Следующий большой переход — от изображений к тексту.

## Навигация

**← Глава 5. MNIST** · **Глава 6. CNN** · **Глава 7 → Текст, токены и Embeddings**
