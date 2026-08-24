---
title: "Глава 11. RAG — свои документы и знания"
project: "Almaz_AI"
chapter: 11
tags: [rag, embeddings, retrieval, vector_search, chunks, llm, qwen]
---

# 📚 Глава 11. RAG — свои документы и знания

> **Главная цель главы:** понять, как дать языковой модели доступ к собственным документам без изменения её весов и построить простой Retrieval-Augmented Generation pipeline.

## 🎯 Что ты изучишь

После этой главы ты сможешь:

- объяснить, что такое RAG;
- понимать разницу между RAG и Fine-tuning;
- понимать Chunks, Chunk Size и Chunk Overlap;
- понимать Embeddings документов;
- понимать Vector Search и Cosine Similarity;
- понимать Top-k Retrieval;
- понимать роль Vector Store и Metadata;
- понимать Retrieval, Augmentation и Generation;
- построить простой локальный RAG pipeline;
- подготовить архитектуру для Local Qwen + собственные документы.

# 1. Проблема, которую решает RAG

Локальная LLM не знает автоматически содержимое ваших файлов:

```text
заметки
PDF
документация
технические задания
таблицы
внутренние инструкции
```

Если документ не передан модели, она не может надёжно отвечать по его содержимому.

![[Images/chapter11/01_rag_problem_solution.png]]

# 2. Что такое RAG

RAG = Retrieval-Augmented Generation.

```text
Вопрос пользователя
↓
поиск релевантных фрагментов
↓
добавление найденного текста в Prompt
↓
LLM
↓
ответ
```

# 3. RAG не обучает weights

Обычный RAG не изменяет веса модели.

```text
RAG = поиск + контекст
```

Нужные знания подаются в Context Window во время запроса.

# 4. RAG и Fine-tuning

```text
RAG
→ даёт внешние знания во время запроса

Fine-tuning
→ изменяет поведение модели через обучение
```

Для ответов по конкретной документации обычно логично сначала пробовать RAG.

# 5. Полный pipeline

Индексация:

```text
Documents
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
Vector Store
```

Запрос:

```text
Question
↓
Question Embedding
↓
Similarity Search
↓
Top-k Chunks
↓
Prompt + Context
↓
LLM
↓
Answer
```

![[Images/chapter11/02_rag_pipeline.png]]

# 6. Что такое Chunk

Chunk — небольшой фрагмент документа.

Например:

```text
абзац
несколько абзацев
300 Tokens
500 символов
```

Он должен быть достаточно маленьким для точного поиска и достаточно большим для сохранения смысла.

# 7. Chunk Size

Слишком маленький Chunk:

```text
теряется контекст
```

Слишком большой:

```text
много лишней информации
хуже точность Retrieval
```

# 8. Chunk Overlap

Соседние Chunks могут перекрываться:

```text
Chunk 1: Tokens 1–300
Chunk 2: Tokens 251–550
```

Overlap:

```text
50 Tokens
```

Это помогает не потерять смысл на границах.

![[Images/chapter11/03_chunking_overlap.png]]

# 9. Embedding документа

Каждый Chunk превращается в числовой вектор:

```text
Chunk Text
↓
Embedding Model
↓
Vector
```

Например:

```text
[0.12, -0.88, 0.31, ...]
```

# 10. Embedding Model и LLM

Embedding Model:

```text
Text → Vector
```

Generative LLM:

```text
Context → Generated Text
```

Это могут быть две разные модели.

# 11. Embedding запроса

```text
Question
↓
Embedding Model
↓
Question Vector
```

Question Vector сравнивается с векторами Chunks.

# 12. Cosine Similarity

Cosine Similarity оценивает близость двух векторов.

Упрощённо:

```text
ближе к 1 → более похожи
около 0   → мало похожи
```

![[Images/chapter11/04_vector_similarity_search.png]]

# 13. Top-k Retrieval

Если в базе 10 000 Chunks, модели не нужны все.

Например:

```text
top_k = 3
```

В Prompt попадут три наиболее релевантных фрагмента.

# 14. Vector Store

Обычно хранят:

```text
Chunk Text
Embedding Vector
Metadata
```

Для этого используют Vector Store или Vector Database.

# 15. Metadata

Полезные поля:

```text
source
filename
page
section
document_id
date
```

Metadata помогает фильтровать результаты и показывать источники.

# 16. Retrieval

```text
Question
↓
Vector Search
↓
Relevant Chunks
```

Это Retrieval.

# 17. Augmentation

Найденные Chunks добавляются в Prompt:

```text
КОНТЕКСТ:
Chunk 1
Chunk 2

ВОПРОС:
...
```

Это Augmentation.

# 18. Generation

LLM получает вопрос и найденный Context и формирует ответ.

```text
Retrieval
+
Augmentation
+
Generation
=
RAG
```

![[Images/chapter11/05_retrieval_augmented_prompt.png]]

# 19. Почему Prompt важен

Полезная инструкция:

```text
Ответь только на основании предоставленного контекста.
Если ответа нет, скажи, что данных недостаточно.
```

Это снижает риск необоснованных дополнений.

# 20. Ограничения RAG

RAG не гарантирует абсолютную точность.

Ошибки могут возникнуть, если:

```text
Retrieval нашёл не тот Chunk
Chunk потерял важный контекст
LLM неверно интерпретировала найденный текст
```

Поэтому полезно показывать Sources.

# 21. Keyword Search и Vector Search

Keyword Search:

```text
ищет совпадения слов
```

Vector Search:

```text
ищет смысловую близость
```

# 22. Hybrid Search

Можно объединить:

```text
Keyword Search
+
Vector Search
```

Это Hybrid Search.

# 23. Reranking — обзорно

Можно получить больше кандидатов:

```text
20 Chunks
↓
Reranker
↓
3 лучших
```

Так можно улучшать качество Retrieval.

# 24. Локальный RAG

```text
Local Documents
↓
Local Embedding Model
↓
Local Vector Store
↓
Local Qwen
↓
Local Answer
```

![[Images/chapter11/06_local_rag_architecture.png]]

# 25. Минимальный RAG без Vector DB

Для учебной лаборатории можно хранить:

```text
список Chunks
+
матрицу Vectors
```

прямо в Python.

# 26. Формирование RAG Prompt

```python
context = "\n\n".join(
    chunk["text"]
    for chunk in retrieved
)

prompt = f"""
Ответь только по контексту.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}
"""
```

# 27. RAG и Context Window

Retrieved Chunks занимают место в Context Window.

Поэтому нужно балансировать:

```text
top_k
chunk_size
```

# 28. RAG и обновляемые знания

Если документ изменился:

```text
обновляем индекс
```

и не обязаны заново Fine-tune модель.

Это удобно для:

```text
документации
регламентов
проектных заметок
актуальных баз знаний
```

# 29. RAG + Agent

```text
Agent
↓
Tool: search_knowledge_base
↓
Retrieval
↓
Observation
↓
LLM
↓
Answer
```

RAG может быть Tool агента.

# 30. 🧪 Что будет в Notebook

```text
Labs/chapter11/11_rag_basics.ipynb
```

Мы:

- создадим маленькую базу документов;
- разделим её на Chunks;
- создадим простой TF-IDF-подобный Vector;
- посчитаем Cosine Similarity;
- сделаем Top-k Retrieval;
- сформируем RAG Prompt;
- покажем Sources;
- создадим `search_knowledge_base`;
- подготовим место для подключения локальной Qwen.

# 31. ❓ Самопроверка

1. Что означает RAG?
2. Изменяет ли RAG weights модели?
3. Что такое Chunk?
4. Зачем Chunk Overlap?
5. Что делает Embedding Model?
6. Чем Embedding Model отличается от LLM?
7. Что такое Cosine Similarity?
8. Что такое Top-k?
9. Что хранится в Vector Store?
10. Зачем Metadata?
11. Что такое Retrieval?
12. Что такое Augmentation?
13. Что такое Generation?
14. Почему Retrieval критичен?
15. Чем Keyword Search отличается от Vector Search?
16. Что такое Hybrid Search?
17. Что такое Reranking?
18. Почему RAG удобен для обновляемых знаний?
19. Можно ли использовать RAG как Tool агента?

# 32. 🧩 Практическое задание

Добавь три своих документа:

```text
project_architecture.txt
rules.txt
notes.txt
```

Затем:

- раздели их на Chunks;
- выполни три запроса;
- посмотри Top-3;
- сравни Scores;
- проверь Sources;
- сформируй RAG Prompt.

# 33. 📌 Что нужно запомнить

Индексация:

```text
Documents
↓
Chunks
↓
Embeddings
↓
Vector Store
```

Запрос:

```text
Question
↓
Question Embedding
↓
Retrieval
↓
Top-k Chunks
↓
Prompt + Context
↓
LLM
↓
Answer
```

Главное различие:

```text
RAG
→ добавляет знания в Context

Fine-tuning
→ обучает параметры / адаптеры
```

# 34. 🚀 Следующая глава

# Глава 12. Fine-tuning, LoRA и QLoRA

Следующим шагом перейдём непосредственно к дообучению модели.

## Навигация

**← Глава 10. AI-агенты и Tools** · **Глава 11. RAG** · **Глава 12 → Fine-tuning, LoRA и QLoRA**
