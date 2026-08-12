---
title: "Глава 4. PyTorch изнутри"
project: "Almaz_AI"
chapter: 4
tags: [pytorch, tensor, nn_module, linear, forward, autograd, parameters]
---

# 🔥 Глава 4. PyTorch изнутри

> **Главная цель главы:** перестать воспринимать PyTorch как набор магических команд и понять, что именно происходит внутри модели: какие объекты создаются, где хранятся параметры, как идёт `forward()`, как PyTorch отслеживает вычисления и как всё это связано с обучением.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить, что такое `Tensor`;
- понимать `shape`, `dtype` и `device`;
- понимать, чем Tensor отличается от обычного списка Python;
- объяснить роль `nn.Module`;
- понимать, что делает `nn.Linear`;
- читать запись `nn.Linear(3, 4)`;
- находить веса и bias внутри слоя;
- понимать, зачем нужен `forward()`;
- понимать разницу между `model(x)` и `model.forward(x)`;
- понимать, что такое `nn.Sequential`;
- использовать `parameters()` и `named_parameters()`;
- понимать `requires_grad`;
- понимать режимы `train()` и `eval()`;
- понимать назначение `torch.no_grad()`;
- понимать роль `autograd`;
- видеть форму Tensor после каждого слоя;
- считать количество обучаемых параметров модели.

# 1. Что такое PyTorch

PyTorch — библиотека для работы с:

```text
Tensor
↓
математическими операциями
↓
нейронными сетями
↓
автоматическим вычислением gradients
↓
обучением моделей
```

# 2. Tensor — базовый объект PyTorch

`Tensor` — многомерный массив чисел.

```python
import torch

values = torch.tensor([1, 2, 3])
```

Tensor может представлять:

```text
одно число
вектор
матрицу
изображение
batch изображений
эмбеддинги
веса нейросети
```

![[Images/chapter04/01_tensor_structure.png]]

# 3. Размерность и shape

```python
x = torch.tensor([1.0, 2.0, 3.0])
print(x.shape)
```

Получим:

```text
torch.Size([3])
```

Для матрицы из 2 строк и 3 столбцов:

```text
shape = [2, 3]
```

# 4. dtype

Примеры:

```text
torch.float32
torch.float64
torch.int64
torch.bool
```

В нейросетях часто используется `torch.float32`.

# 5. device — CPU или GPU

Tensor находится на определённом устройстве.

```python
print(x.device)
```

Например:

```text
cpu
```

При наличии CUDA:

```python
x = x.to("cuda")
```

# 6. Tensor и Python list

Python list — универсальный контейнер.

Tensor — специализированный числовой объект с поддержкой:

```text
быстрых математических операций
CPU/GPU
autograd
нейросетей
```

# 7. Что такое nn.Module

Большинство моделей PyTorch наследуются от:

```python
nn.Module
```

Он даёт:

```text
регистрацию слоёв
регистрацию параметров
parameters()
named_parameters()
train()
eval()
state_dict()
```

![[Images/chapter04/02_nn_module.png]]

# 8. Что такое nn.Linear

`nn.Linear` выполняет линейное преобразование:

\[
y = xW^T + b
\]

Запись:

```python
nn.Linear(3, 4)
```

означает:

```text
3 входных признака
↓
4 выходных значения
```

![[Images/chapter04/03_linear_layer.png]]

# 9. Что хранится внутри Linear

У `nn.Linear(3, 4)` есть:

```text
weight shape = [4, 3]
bias shape   = [4]
```

Количество параметров:

```text
12 weights + 4 bias = 16
```

# 10. Что такое forward()

```python
def forward(self, x):
    x = self.layer1(x)
    x = self.relu(x)
    x = self.layer2(x)
    return x
```

`forward()` описывает путь данных через модель.

![[Images/chapter04/04_forward_pass_pytorch.png]]

# 11. model(x) и model.forward(x)

Обычно используется:

```python
output = model(x)
```

а не прямой вызов:

```python
model.forward(x)
```

`model(x)` проходит через внутреннюю механику `nn.Module` и уже вызывает `forward()`.

# 12. nn.Sequential

Для простой цепочки удобно:

```python
nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
```

# 13. parameters() и named_parameters()

```python
for parameter in model.parameters():
    print(parameter)
```

или:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

![[Images/chapter04/05_parameters_autograd.png]]

# 14. requires_grad

Если:

```python
parameter.requires_grad == True
```

PyTorch отслеживает операции, чтобы потом можно было вычислить gradient.

# 15. parameter.grad

До `backward()`:

```text
parameter.grad = None
```

После:

```python
loss.backward()
```

в `.grad` появляются gradients.

# 16. autograd

`autograd` — система автоматического дифференцирования PyTorch.

Она отслеживает математические операции и строит вычислительный граф.

# 17. train() и eval()

```python
model.train()
model.eval()
```

`train()` — режим обучения.

`eval()` — режим оценки/inference.

Особенно важно для слоёв вроде `Dropout` и `BatchNorm`.

# 18. torch.no_grad()

Для inference:

```python
with torch.no_grad():
    output = model(x)
```

В этом блоке PyTorch не строит граф для gradients.

# 19. Как распечатать архитектуру

```python
print(model)
```

# 20. Как посчитать параметры

```python
total = sum(
    parameter.numel()
    for parameter in model.parameters()
)
```

Для нашей сети:

```text
Linear(3,4) = 16 параметров
Linear(4,1) = 5 параметров
Итого = 21
```

# 21. Снова разбираем сеть «Светофор»

```text
Input: 3
↓
Linear(3 → 4)
↓
ReLU
↓
Linear(4 → 1)
↓
Logit
```

# 22. Tensor на каждом этапе

Один объект:

```python
x = torch.tensor([[0.0, 0.0, 1.0]])
```

Shape:

```text
[1, 3]
```

После `Linear(3,4)`:

```text
[1, 4]
```

После ReLU:

```text
[1, 4]
```

После `Linear(4,1)`:

```text
[1, 1]
```

![[Images/chapter04/06_pytorch_model_inside.png]]

# 23. Почему Shape так важен

Если слой:

```python
nn.Linear(3, 4)
```

ожидает 3 входных признака, а получает 5 — будет ошибка размерности.

Очень многие ошибки в ML связаны именно с неправильными Shapes.

# 24. 🌍 Аналогия

```text
сырьё
↓
станок
↓
промежуточный результат
↓
станок
↓
готовый продукт
```

В PyTorch:

```text
Tensor
↓
Layer
↓
Tensor
↓
Activation
↓
Tensor
↓
Layer
↓
Tensor
```

# 25. 🧪 Что будет в Notebook

Notebook:

```text
Labs/chapter04/04_pytorch_inside.ipynb
```

Мы:

- создадим Tensor;
- посмотрим `shape`, `dtype`, `device`;
- создадим `nn.Linear`;
- посмотрим weights и bias;
- создадим `nn.Module`;
- изучим `forward()`;
- посмотрим параметры;
- проверим `requires_grad`;
- вызовем `backward()`;
- посмотрим `.grad`;
- разберём `train()` / `eval()`;
- используем `torch.no_grad()`;
- пройдём по модели слой за слоем.

# 26. ❓ Самопроверка

1. Что такое Tensor?
2. Что показывает `shape`?
3. Что такое `dtype`?
4. Что показывает `device`?
5. Почему модели наследуются от `nn.Module`?
6. Что означает `nn.Linear(3,4)`?
7. Какой Shape у его weight?
8. Что делает `forward()`?
9. Почему обычно используют `model(x)`?
10. Что такое `nn.Sequential`?
11. Что возвращает `named_parameters()`?
12. Что означает `requires_grad=True`?
13. Что такое `autograd`?
14. Что хранится в `.grad`?
15. Для чего `train()` и `eval()`?
16. Для чего нужен `torch.no_grad()`?
17. Почему важно следить за Shape?

# 27. 🧩 Практическое задание

Создай сеть:

```text
4 входа
↓
Linear(4 → 8)
↓
ReLU
↓
Linear(8 → 2)
```

Определи:

```text
Shape первого weight
количество bias первого слоя
Shape после первого Linear
Shape после ReLU
Shape выхода
общее количество параметров
```

# 28. 📌 Что нужно запомнить

```text
Tensor
↓
nn.Module
↓
Layers
↓
forward()
↓
Tensor
```

Основные инструменты:

```python
tensor.shape
tensor.dtype
tensor.device

model.parameters()
model.named_parameters()

parameter.requires_grad
parameter.grad

model.train()
model.eval()

torch.no_grad()
```

# 29. 🚀 Переход к главе 5

# Глава 5. MNIST — распознавание рукописных цифр

Следующим шагом перейдём от искусственного Dataset к настоящим изображениям цифр.

## Навигация

**← Глава 3. Loss, Gradient Descent и Backpropagation** · **Глава 4. PyTorch изнутри** · **Глава 5 → MNIST**
