---
title: "Глава 7. Текст, токены и Embeddings"
project: "Almaz_AI"
chapter: 7
tags: [text, tokens, tokenizer, vocabulary, embeddings, pytorch, nlp]
---

# 📝 Глава 7. Текст, токены и Embeddings

> **Главная цель главы:** понять, как обычный текст превращается в числа, которые уже может обработать нейронная сеть.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить, почему нейросеть не работает напрямую со строкой текста;
- понимать, что такое Token;
- понимать разницу между словом, символом и subword-токеном;
- понимать, что такое Tokenizer;
- понимать Vocabulary;
- понимать Token ID;
- понимать Special Tokens;
- понимать Padding;
- понимать Attention Mask на базовом уровне;
- понимать, что делает `nn.Embedding`;
- понимать Embedding Matrix;
- понимать Shape последовательности токенов;
- превращать простое предложение в Token IDs;
- получать Embedding-векторы для токенов;
- понимать, почему Embeddings являются фундаментом LLM.

# 1. Почему текст нужно превращать в числа

Нейронная сеть работает с Tensor, а не со строками.

```text
Text
↓
Tokenizer
↓
Tokens
↓
Token IDs
↓
Embeddings
↓
Tensor
```

![[Images/chapter07/01_text_to_tokens.png]]

# 2. Что такое Token

Token — элемент текста, с которым работает Tokenizer.

Простой пример:

```text
"Я изучаю ИИ"
↓
["Я", "изучаю", "ИИ"]
```

Но Token не обязан быть целым словом.

# 3. Три подхода к токенизации

### Word-level

```text
"кот сидит дома"
↓
["кот", "сидит", "дома"]
```

### Character-level

```text
"кот"
↓
["к", "о", "т"]
```

### Subword

Условно:

```text
"нейросетями"
↓
["нейро", "сет", "ями"]
```

Реальный Tokenizer конкретной модели может разбить слово иначе.

![[Images/chapter07/02_tokenization_types.png]]

# 4. Что такое Tokenizer

Tokenizer:

```text
получает Text
↓
разбивает на Tokens
↓
сопоставляет Tokens с Token IDs
```

У разных моделей могут быть разные Tokenizer.

# 5. Vocabulary

Vocabulary — словарь токенов модели.

Например:

```text
"<PAD>" → 0
"<UNK>" → 1
"я"     → 2
"люблю" → 3
"python"→ 4
```

# 6. Token IDs

```text
"я люблю python"
↓
["я", "люблю", "python"]
↓
[2, 3, 4]
```

Теперь это можно представить Tensor:

```python
torch.tensor([2, 3, 4])
```

![[Images/chapter07/03_vocabulary_token_ids.png]]

# 7. Special Tokens

Частые специальные токены:

```text
<PAD>
<UNK>
<BOS>
<EOS>
```

У разных моделей обозначения могут отличаться.

# 8. Padding

Предложения имеют разную длину.

Например:

```text
A → [2, 3, 4]
B → [2, 7, 8, 9, 10]
```

После Padding:

```text
A → [2, 3, 4, 0, 0]
B → [2, 7, 8, 9, 10]
```

# 9. Attention Mask

Упрощённо:

```text
Token IDs:
[2, 3, 4, 0, 0]

Attention Mask:
[1, 1, 1, 0, 0]
```

Где `1` — настоящий Token, `0` — Padding.

![[Images/chapter07/04_padding_attention_mask.png]]

# 10. Почему Token ID недостаточен

Token ID — просто номер.

```text
кот → 25
собака → 781
```

Числа 25 и 781 сами по себе не содержат смысла и не показывают близость понятий.

Нужен Embedding.

# 11. Что такое Embedding

Embedding превращает Token ID в вектор чисел.

```text
Token ID 25
↓
[0.12, -0.41, 0.88, 0.03]
```

Другой Token получает другой вектор.

![[Images/chapter07/05_embedding_lookup.png]]

# 12. nn.Embedding

```python
embedding = nn.Embedding(
    num_embeddings=1000,
    embedding_dim=64,
)
```

Это означает:

```text
Vocabulary size = 1000
Embedding dimension = 64
```

Embedding Matrix имеет Shape:

```text
[1000, 64]
```

# 13. Embedding Lookup

Вход:

```text
[2, 3, 4]
```

При `embedding_dim=4`:

```text
input  → [3]
output → [3, 4]
```

# 14. Batch Embeddings

Если Token IDs имеют Shape:

```text
[32, 20]
```

где:

```text
32 → Batch
20 → Sequence Length
```

и `embedding_dim=128`, после Embedding:

```text
[32, 20, 128]
```

![[Images/chapter07/06_text_embedding_pipeline.png]]

# 15. Embeddings обучаются

`nn.Embedding` содержит обучаемые параметры.

```text
Loss
↓
Backward
↓
Gradient
↓
Optimizer
↓
Embedding vectors обновляются
```

# 16. Семантическая идея

После обучения похожие понятия могут получать векторы, которые определённым образом близки.

Например:

```text
кот
собака
животное
```

могут оказаться ближе друг к другу, чем:

```text
кот
экскаватор
```

Но случайно созданный `nn.Embedding` такого смысла ещё не имеет.

# 17. Проблема порядка слов

После Embedding модель получила последовательность векторов.

Но ей ещё нужно понимать порядок:

```text
кот укусил собаку
```

не равно:

```text
собака укусила кота
```

Это приведёт нас к позиционной информации и Transformer.

# 18. Как это связано с LLM

Упрощённо:

```text
текст
↓
Tokenizer
↓
Token IDs
↓
Embedding
↓
Transformer Blocks
↓
Logits
↓
следующий Token
```

# 19. 🧪 Что будет в Notebook

```text
Labs/chapter07/07_text_tokens_embeddings.ipynb
```

Мы:

- создадим маленький текстовый Dataset;
- построим Vocabulary;
- токенизируем предложения;
- получим Token IDs;
- добавим `<UNK>` и `<PAD>`;
- сделаем Padding;
- построим Attention Mask;
- создадим `nn.Embedding`;
- посмотрим Embedding Matrix;
- получим Embeddings;
- посмотрим Shapes для Batch;
- проверим, что Embeddings обучаемые.

# 20. ❓ Самопроверка

1. Почему нейросеть не работает напрямую со строкой?
2. Что такое Token?
3. Обязан ли Token быть словом?
4. Что такое Tokenizer?
5. Что такое Vocabulary?
6. Что такое Token ID?
7. Зачем `<UNK>`?
8. Зачем Padding?
9. Что показывает Attention Mask?
10. Почему Token ID не является смысловым числом?
11. Что делает Embedding?
12. Что означает `nn.Embedding(1000,64)`?
13. Какой Shape у Embedding Matrix?
14. Какой Shape получится из `[32,20]` при `embedding_dim=128`?
15. Обучаются ли Embeddings?
16. Почему модели нужно знать порядок Tokens?

# 21. 🧩 Практическое задание

Создай Vocabulary:

```text
<PAD>
<UNK>
я
люблю
python
изучаю
нейросети
```

Закодируй:

```text
я люблю python
я изучаю нейросети
я люблю transformer
```

Неизвестное слово замени на `<UNK>`.

Затем:

- сделай Padding;
- создай Attention Mask;
- создай `nn.Embedding`;
- получи Shape выходного Tensor.

# 22. 📌 Что нужно запомнить

```text
Text
↓
Tokenizer
↓
Tokens
↓
Vocabulary
↓
Token IDs
↓
Padding / Mask
↓
Embedding
↓
Vector Sequence
```

Ключевой Shape:

```text
[Batch, Sequence Length]
↓ Embedding
[Batch, Sequence Length, Embedding Dim]
```

# 23. 🚀 Следующая глава

# Глава 8. Transformer и Attention

## Навигация

**← Глава 6. CNN** · **Глава 7. Текст, токены и Embeddings** · **Глава 8 → Transformer и Attention**
