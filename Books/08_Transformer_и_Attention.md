---
title: "Глава 8. Transformer и Attention"
project: "Almaz_AI"
chapter: 8
tags: [transformer, attention, self_attention, qkv, softmax, positional_encoding, llm]
---

# 🤖 Глава 8. Transformer и Attention

> **Главная цель главы:** понять центральную идею современных языковых моделей — механизм Attention — и увидеть, как Transformer обрабатывает последовательность токенов.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить, зачем языковой модели нужен Attention;
- понимать, что такое Self-Attention;
- понимать роли `Query`, `Key`, `Value`;
- понимать, как вычисляются Attention Scores;
- понимать роль Softmax;
- понимать Weighted Sum;
- понимать, что такое Attention Matrix;
- понимать Multi-Head Attention;
- понимать Positional Information;
- понимать Residual Connections;
- понимать Layer Normalization;
- понимать Feed-Forward Network внутри Transformer;
- понимать Causal Mask;
- понимать базовую структуру Transformer Block;
- понимать, как Transformer связан с LLM.

# 1. Почему Embeddings недостаточно

В прошлой главе мы получили:

```text
Text
↓
Tokenizer
↓
Token IDs
↓
Embeddings
```

Но модель ещё должна понять:

> Какие другие Tokens важны для текущего Token?

Например:

```text
кот сидит на ковре потому что он устал
```

Слово `он` должно учитывать связь со словом `кот`.

![[Images/chapter08/01_attention_intuition.png]]

# 2. Что такое Attention

Attention — механизм, который позволяет Token определить, на какие другие Tokens нужно обратить больше внимания.

Условный пример:

```text
Token: "он"

кот       → 0.70
сидит     → 0.05
ковре     → 0.03
потому    → 0.07
устал     → 0.15
```

# 3. Self-Attention

Если Tokens одной последовательности обращают внимание друг на друга:

```text
Self-Attention
```

Одна последовательность используется для построения:

```text
Queries
Keys
Values
```

# 4. Query, Key и Value

Для каждого Token вычисляются три вектора:

```text
Q = Query
K = Key
V = Value
```

Интуитивно:

```text
Query → что я ищу?
Key   → что я могу предложить?
Value → какую информацию я передам?
```

![[Images/chapter08/02_query_key_value.png]]

# 5. Откуда берутся Q, K, V

Для Embedding `x`:

\[
Q = xW_Q
\]

\[
K = xW_K
\]

\[
V = xW_V
\]

`W_Q`, `W_K`, `W_V` — обучаемые матрицы.

# 6. Attention Score

Query одного Token сравнивается с Key другого:

\[
score = QK^T
\]

Чем выше Score, тем сильнее потенциальная связь.

# 7. Главная формула

\[
Attention(Q,K,V)=
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
\]

![[Images/chapter08/03_attention_formula.png]]

# 8. Шаги Attention

```text
Q × Kᵀ
↓
Scores
↓
Scale
↓
Softmax
↓
Attention Weights
↓
Weights × V
↓
Contextual Representation
```

# 9. Почему нужен Scale

Если размерность Key большая, скалярные произведения становятся крупными.

Поэтому:

\[
QK^T
\]

делится на:

\[
\sqrt{d_k}
\]

Это делает обучение стабильнее.

# 10. Softmax

Softmax превращает Scores в веса.

Например:

```text
[2.1, 0.5, -0.3]
↓
[0.76, 0.15, 0.09]
```

Сумма весов:

```text
1
```

# 11. Weighted Sum

Attention Weights используются для объединения Values.

Token получает смесь информации от других Tokens.

# 12. Attention Matrix

Если Sequence Length = 5:

```text
Attention Matrix = [5,5]
```

Строка — текущий Query Token.

Столбец — Token, на который он смотрит.

![[Images/chapter08/04_attention_matrix.png]]

# 13. Multi-Head Attention

Одна Attention Head может изучать один тип связей.

Несколько Heads позволяют параллельно учитывать:

```text
грамматику
смысл
локальные связи
дальние связи
ссылки на объекты
```

![[Images/chapter08/05_multihead_attention.png]]

# 14. Positional Information

Attention сам по себе не знает порядок Tokens.

Поэтому модель получает позиционную информацию.

Например:

```text
Token 0
Token 1
Token 2
...
```

Конкретная реализация зависит от архитектуры модели.

# 15. Residual Connection

Упрощённо:

```text
Output = Block(x) + x
```

Исходный сигнал добавляется обратно.

Это помогает глубоким сетям обучаться устойчивее.

# 16. LayerNorm

Layer Normalization помогает стабилизировать внутренние представления.

# 17. Feed-Forward Network

После Attention каждый Token проходит через небольшую MLP:

```text
Linear
↓
Activation
↓
Linear
```

# 18. Transformer Block

Упрощённо:

```text
Input
↓
Self-Attention
↓
Residual
↓
LayerNorm
↓
Feed-Forward
↓
Residual
↓
LayerNorm
```

В реальных архитектурах порядок LayerNorm может отличаться.

![[Images/chapter08/06_transformer_block.png]]

# 19. Causal Mask

Генеративная модель не должна видеть будущие Tokens.

Упрощённо:

```text
Token 1 видит 1
Token 2 видит 1–2
Token 3 видит 1–3
...
```

Это называется:

```text
Causal Mask
```

# 20. Padding Mask и Causal Mask

Padding Mask:

```text
не учитывать PAD
```

Causal Mask:

```text
не смотреть в будущее
```

Это разные задачи.

# 21. Shapes

Пусть:

```text
Batch = 2
Sequence = 5
Embedding Dim = 8
```

Вход:

```text
[2,5,8]
```

Q, K, V:

```text
[2,5,8]
```

Attention Scores:

```text
[2,5,5]
```

Output:

```text
[2,5,8]
```

# 22. Минимальный Self-Attention

```python
Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

scores = Q @ K.transpose(-2, -1)
scores = scores / sqrt(d_k)

weights = softmax(scores)
output = weights @ V
```

# 23. nn.MultiheadAttention

PyTorch имеет готовый слой:

```python
nn.MultiheadAttention(
    embed_dim=64,
    num_heads=4,
    batch_first=True,
)
```

# 24. Как Transformer связан с LLM

```text
Text
↓
Tokens
↓
Token IDs
↓
Embeddings
↓
Positional Information
↓
Transformer Blocks
↓
Contextual Representations
↓
Logits
↓
Next Token
```

# 25. Contextual Representation

Слово:

```text
банк
```

в:

```text
банк выдал кредит
```

и:

```text
банк реки
```

может получить разные представления благодаря контексту.

# 26. 🧪 Что будет в Notebook

```text
Labs/chapter08/08_transformer_attention.ipynb
```

Мы:

- создадим Embeddings;
- построим Q, K, V;
- вычислим Scores;
- применим Scale;
- применим Softmax;
- получим Weighted Sum;
- визуализируем Attention Matrix;
- создадим Causal Mask;
- используем `nn.MultiheadAttention`;
- соберём маленький Transformer Block.

# 27. ❓ Самопроверка

1. Зачем нужен Attention?
2. Что такое Self-Attention?
3. Что означает Query?
4. Что означает Key?
5. Что означает Value?
6. Откуда берутся Q, K, V?
7. Как считается Score?
8. Зачем Scale Factor?
9. Зачем Softmax?
10. Что такое Attention Matrix?
11. Что такое Multi-Head Attention?
12. Зачем нужна позиционная информация?
13. Что такое Residual Connection?
14. Для чего LayerNorm?
15. Что делает Feed-Forward часть?
16. Что такое Causal Mask?
17. Чем Causal Mask отличается от Padding Mask?
18. Почему представление Token после Attention становится контекстным?

# 28. 📌 Что нужно запомнить

```text
Embeddings
↓
Q, K, V
↓
QKᵀ
↓
Scale
↓
Softmax
↓
Attention Weights
↓
Weighted Sum of V
↓
Contextual Representations
```

# 29. 🚀 Следующая глава

# Глава 9. Локальные LLM и Qwen

## Навигация

**← Глава 7. Текст, токены и Embeddings** · **Глава 8. Transformer и Attention** · **Глава 9 → Локальные LLM**
