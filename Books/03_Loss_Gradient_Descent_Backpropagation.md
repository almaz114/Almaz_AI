---
title: "Глава 3. Loss, Gradient Descent и Backpropagation"
project: "Almaz_AI"
chapter: 3
tags: [нейронные_сети, loss, gradient, backpropagation, optimizer, pytorch]
---

# 🧠 Глава 3. Loss, Gradient Descent и Backpropagation

> **Главная цель главы:** понять, как нейросеть обнаруживает свою ошибку, как вычисляет направление изменения параметров и как шаг за шагом обучается.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить разницу между `Prediction` и `Target`;
- понимать, что такое `Loss Function`;
- отличать `Loss` от `Accuracy`;
- понимать идею `MSE` и `Binary Cross Entropy`;
- понимать, зачем в PyTorch нужен `BCEWithLogitsLoss`;
- понимать идею производной без углубления в сложный анализ;
- объяснить, что такое `Gradient`;
- понимать `Gradient Descent`;
- объяснить роль `Learning Rate`;
- понимать, что такое `Backpropagation`;
- объяснить, что делают `loss.backward()`, `optimizer.step()` и `optimizer.zero_grad()`;
- смотреть значения `.grad` у параметров модели;
- проследить один шаг обучения буквально по числам;
- понимать полный цикл обучения нейросети.

# 1. Главная идея обучения

```text
Входные данные
      ↓
Forward Pass
      ↓
Prediction
      ↓
Сравнение с Target
      ↓
Loss
      ↓
Gradient
      ↓
Backpropagation
      ↓
Optimizer
      ↓
Изменение весов
      ↓
Новый Forward Pass
```

![[Images/chapter03/01_prediction_vs_target.png]]

# 2. Prediction и Target

Пусть модель должна ответить:

```text
0 = STOP
1 = GO
```

Правильный ответ:

```text
Target = 1
```

Модель предсказала:

```text
Prediction = 0.32
```

Модели нужно выразить величину ошибки числом. Для этого нужна функция потерь.

# 3. Что такое Loss Function

`Loss Function` измеряет ошибку модели.

```text
Prediction
     +
Target
     ↓
Loss Function
     ↓
Loss
```

![[Images/chapter03/02_loss_function.png]]

# 4. Loss и Accuracy — не одно и то же

`Accuracy` показывает долю правильных ответов.

`Loss` показывает величину ошибки с точки зрения выбранной математической функции.

Две модели могут иметь одинаковую Accuracy, но разный Loss.

# 5. MSE — Mean Squared Error

Для одного значения:

\[
Loss = (Prediction - Target)^2
\]

Пример:

```text
Prediction = 0.7
Target = 1.0
Loss = 0.09
```

В PyTorch:

```python
nn.MSELoss()
```

MSE часто используется в задачах регрессии.

# 6. Binary Cross Entropy

Для бинарной классификации часто используется BCE.

Примеры задач:

```text
0 / 1
STOP / GO
да / нет
спам / не спам
```

В нашей задаче «Светофор» используется:

```python
nn.BCEWithLogitsLoss()
```

# 7. Почему BCEWithLogitsLoss работает с logits

Наша модель обычно выдаёт сырое число — `logit`.

Чтобы превратить его в вероятность, можно использовать:

```python
torch.sigmoid(logit)
```

Но во время обучения правильно делать так:

```python
logits = model(x)
loss = loss_function(logits, target)
```

`BCEWithLogitsLoss` уже объединяет необходимые вычисления численно устойчивым способом.

# 8. Как уменьшить Loss

Если мы получили Loss, нужно понять, какие веса изменить.

Допустим:

```text
weight = 0.42
Loss = 0.78
```

Если немного изменить вес, Loss тоже изменится. Нам важно понять:

```text
в какую сторону?
насколько сильно?
```

Здесь появляется градиент.

# 9. Интуитивное понятие производной

Производная отвечает на вопрос:

> Если я немного изменю вход, как изменится результат?

Для нейросети:

```text
немного изменили weight
        ↓
Loss изменился
```

# 10. Что такое Gradient

Gradient показывает, как изменится Loss при изменении параметров.

Если gradient положительный, увеличение параметра обычно увеличивает Loss.

Если отрицательный — увеличение параметра уменьшает Loss.

![[Images/chapter03/03_gradient_intuition.png]]

# 11. Gradient Descent

Чтобы уменьшить Loss, мы идём против направления градиента:

\[
w_{new} = w_{old} - learning\_rate \cdot gradient
\]

![[Images/chapter03/04_gradient_descent.png]]

# 12. Learning Rate

`learning_rate` определяет размер шага.

```text
слишком маленький → обучение медленное
разумный → стабильное движение
слишком большой → возможные скачки и расходимость
```

# 13. Что такое Backpropagation

`Backpropagation` распространяет информацию об ошибке назад по вычислительному графу и вычисляет градиенты параметров.

```text
Forward:
Input → Layer → Activation → Layer → Prediction

Backward:
Loss → Layer → Activation → Layer
```

![[Images/chapter03/05_backpropagation.png]]

# 14. Что делает loss.backward()

```python
loss.backward()
```

вычисляет градиенты Loss по параметрам модели.

После этого они доступны через:

```python
parameter.grad
```

# 15. Что делает optimizer.step()

После расчёта градиентов:

```python
optimizer.step()
```

обновляет параметры модели.

# 16. Зачем optimizer.zero_grad()

В PyTorch градиенты накапливаются.

Поэтому перед очередным `backward()` обычно вызывается:

```python
optimizer.zero_grad()
```

# 17. Полный цикл обучения

```python
for epoch in range(epochs):
    logits = model(features)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

![[Images/chapter03/06_training_cycle.png]]

# 18. Один шаг обучения под микроскопом

```text
Вес ДО
w = 0.4200

↓ Forward Pass

Prediction

↓ Loss

↓ backward()

Gradient
w.grad = -0.1800

↓ optimizer.step()

Вес ПОСЛЕ
w = 0.4312
```

Это главный практический эксперимент главы.

# 19. Возвращаемся к сети «Светофор»

Используем архитектуру:

```text
3 входа
   ↓
Linear(3 → 4)
   ↓
ReLU
   ↓
Linear(4 → 1)
```

Функция потерь:

```python
nn.BCEWithLogitsLoss()
```

Optimizer:

```python
torch.optim.Adam(...)
```

# 20. Что такое Epoch

Одна эпоха — один полный проход по обучающему набору данных.

В нашем маленьком датасете одна эпоха означает обработку всех 8 состояний светофора.

# 21. Почему Loss визуализируют

График Loss помогает увидеть:

- идёт ли обучение;
- уменьшается ли ошибка;
- не слишком ли большой Learning Rate;
- не остановилось ли обучение.

# 22. Эксперимент с Learning Rate

В Notebook сравним:

```text
0.001
0.01
0.05
0.5
```

и построим кривые Loss.

# 23. MSE и BCE — базовая карта

| Задача | Частый Loss |
|---|---|
| Предсказание числа | MSE |
| Бинарная классификация | BCE / BCEWithLogitsLoss |
| Несколько классов | CrossEntropyLoss |

# 24. Optimizer

Optimizer — это алгоритм обновления параметров.

Примеры:

```text
SGD
Adam
AdamW
```

# 25. Вся система обучения

```text
DATASET
   ↓
MODEL
   ↓
PREDICTION
   ↓
LOSS FUNCTION
   ↓
LOSS
   ↓
BACKPROPAGATION
   ↓
GRADIENTS
   ↓
OPTIMIZER
   ↓
UPDATED PARAMETERS
```

# 26. 🌍 Аналогия из жизни

Человек учится бросать мяч в корзину:

```text
попытка
↓
ошибка
↓
коррекция
↓
новая попытка
```

Нейросеть делает похожий цикл, только корректирует параметры.

# 27. 🧪 Что будет в лабораторной работе

Notebook:

```text
Labs/chapter03/03_loss_gradient_backpropagation.ipynb
```

Мы:

1. посчитаем MSE вручную;
2. посмотрим BCEWithLogitsLoss;
3. создадим маленькую сеть;
4. посмотрим параметры ДО;
5. сделаем Forward Pass;
6. получим Loss;
7. выполним `backward()`;
8. посмотрим `.grad`;
9. выполним `optimizer.step()`;
10. сравним параметры ДО и ПОСЛЕ;
11. запустим полное обучение;
12. построим Loss curve;
13. сравним Learning Rate.

# 28. ❓ Самопроверка

1. Что такое Prediction?
2. Что такое Target?
3. Для чего нужна Loss Function?
4. Чем Loss отличается от Accuracy?
5. Что измеряет MSE?
6. Для какого типа задач часто используется BCE?
7. Что такое logit?
8. Почему перед BCEWithLogitsLoss не нужен Sigmoid?
9. Что такое Gradient?
10. Почему Gradient Descent движется против Gradient?
11. Что такое Learning Rate?
12. Что делает `loss.backward()`?
13. Где хранится gradient параметра?
14. Что делает `optimizer.step()`?
15. Почему нужен `optimizer.zero_grad()`?
16. Что такое Epoch?

# 29. 🧩 Практическое задание

Дано:

```text
weight = 2.0
gradient = 0.4
learning_rate = 0.1
```

Используй:

\[
w_{new}=w_{old}-learning\_rate\cdot gradient
\]

Затем повтори вычисление для:

```text
gradient = -0.4
```

# 30. 📌 Что нужно запомнить

```text
Prediction
   ↓
Loss
   ↓
Gradient
   ↓
Backpropagation
   ↓
Optimizer
   ↓
Updated Weights
```

Ключевые команды:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

# 31. 🚀 Переход к главе 4

Следующая глава:

# Глава 4. PyTorch изнутри

Разберём:

```text
Tensor
nn.Module
nn.Linear
forward()
parameters()
nn.Sequential
train()
eval()
```

## Навигация

**← Глава 2. Функции активации** · **Глава 3. Loss, Gradient Descent и Backpropagation** · **Глава 4 → PyTorch изнутри**
