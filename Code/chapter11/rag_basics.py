"""Глава 11: RAG — свои документы и знания.

Чистый Python-пример проекта Almaz_AI.

Учебная реализация:
- документы;
- Chunking;
- TF-IDF-подобная векторизация;
- Cosine Similarity;
- Top-k Retrieval;
- RAG Prompt;
- search_knowledge_base как Tool.

Для понимания архитектуры внешняя Embedding Model не требуется.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import torch


TOP_K = 3
CHUNK_SIZE = 12
CHUNK_OVERLAP = 3


DOCUMENTS = [
    {
        "source": "architecture.md",
        "text": (
            "Основной торговый алгоритм использует "
            "Magic Number 112. Позиции и отложенные "
            "ордера Aspid используют Magic Number 115."
        ),
    },
    {
        "source": "license.md",
        "text": (
            "Проверка лицензии выполняется через API. "
            "При сетевой ошибке торговая логика "
            "может продолжить работу."
        ),
    },
    {
        "source": "aspid.md",
        "text": (
            "Для работы Aspid требуется hedging-режим "
            "торгового счёта. Проверка режима "
            "выполняется один раз при запуске."
        ),
    },
]


def tokenize(text: str) -> list[str]:
    """Простейшая токенизация текста."""

    return re.findall(
        r"[а-яёa-z0-9]+",
        text.lower(),
    )


def chunk_words(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Разбивает текст на перекрывающиеся Chunks."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size должен быть > 0"
        )

    if overlap < 0:
        raise ValueError(
            "overlap не может быть отрицательным"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap должен быть меньше chunk_size"
        )

    words = tokenize(text)
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if chunk:
            chunks.append(
                " ".join(chunk)
            )

        if end >= len(words):
            break

        start += chunk_size - overlap

    return chunks


class SimpleRAG:
    """Минимальный локальный Retrieval pipeline."""

    def __init__(
        self,
        documents: list[dict[str, str]],
    ) -> None:
        self.documents = documents
        self.chunks = self._build_chunks()
        self.vocabulary = self._build_vocabulary()

        self.token_to_id = {
            token: index
            for index, token in enumerate(self.vocabulary)
        }

        self.document_frequency = (
            self._build_document_frequency()
        )

        self.chunk_vectors = torch.stack(
            [
                self.text_to_vector(
                    chunk["text"]
                )
                for chunk in self.chunks
            ]
        )

    def _build_chunks(
        self,
    ) -> list[dict[str, str]]:
        """Создаёт Chunks с Metadata."""

        chunks: list[dict[str, str]] = []

        for document in self.documents:
            for index, text in enumerate(
                chunk_words(
                    document["text"]
                )
            ):
                chunks.append(
                    {
                        "id": (
                            f"{document['source']}::{index}"
                        ),
                        "source": document["source"],
                        "text": text,
                    }
                )

        return chunks

    def _build_vocabulary(
        self,
    ) -> list[str]:
        """Создаёт Vocabulary."""

        return sorted(
            {
                token
                for chunk in self.chunks
                for token in tokenize(
                    chunk["text"]
                )
            }
        )

    def _build_document_frequency(
        self,
    ) -> Counter[str]:
        """Считает число Chunks, содержащих Token."""

        frequencies: Counter[str] = Counter()

        for chunk in self.chunks:
            for token in set(
                tokenize(chunk["text"])
            ):
                frequencies[token] += 1

        return frequencies

    def text_to_vector(
        self,
        text: str,
    ) -> torch.Tensor:
        """Преобразует текст в TF-IDF-подобный Vector."""

        tokens = tokenize(text)
        counts = Counter(tokens)

        vector = torch.zeros(
            len(self.vocabulary),
            dtype=torch.float32,
        )

        total_chunks = len(self.chunks)

        for token, count in counts.items():
            token_id = self.token_to_id.get(
                token
            )

            if token_id is None:
                continue

            tf = count / max(
                len(tokens),
                1,
            )

            df = self.document_frequency.get(
                token,
                0,
            )

            idf = math.log(
                (total_chunks + 1)
                / (df + 1)
            ) + 1.0

            vector[token_id] = tf * idf

        return vector

    @staticmethod
    def cosine_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Возвращает Cosine Similarity."""

        denominator = (
            torch.linalg.vector_norm(a)
            * torch.linalg.vector_norm(b)
        )

        if denominator.item() == 0:
            return 0.0

        return float(
            torch.dot(a, b)
            / denominator
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """Возвращает наиболее релевантные Chunks."""

        question_vector = self.text_to_vector(
            question
        )

        results: list[dict[str, Any]] = []

        for chunk, vector in zip(
            self.chunks,
            self.chunk_vectors,
        ):
            score = self.cosine_similarity(
                question_vector,
                vector,
            )

            results.append(
                {
                    **chunk,
                    "score": score,
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]

    def build_prompt(
        self,
        question: str,
        *,
        top_k: int = TOP_K,
    ) -> str:
        """Создаёт Prompt с Retrieved Context."""

        results = self.retrieve(
            question,
            top_k=top_k,
        )

        context = "\n\n".join(
            (
                f"[Источник: {item['source']}]\n"
                f"{item['text']}"
            )
            for item in results
        )

        return (
            "Ответь только на основании контекста.\n"
            "Если ответа нет, скажи, что данных недостаточно.\n\n"
            f"КОНТЕКСТ:\n{context}\n\n"
            f"ВОПРОС:\n{question}"
        )


def search_knowledge_base(
    rag: SimpleRAG,
    question: str,
    *,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Read-only Tool для будущего AI-агента."""

    return {
        "status": "ok",
        "query": question,
        "results": rag.retrieve(
            question,
            top_k=top_k,
        ),
    }


def main() -> None:
    """Запускает учебный RAG pipeline."""

    rag = SimpleRAG(
        DOCUMENTS
    )

    questions = [
        (
            "Какой Magic Number используется "
            "основным алгоритмом?"
        ),
        (
            "Какой режим счёта нужен "
            "для Aspid?"
        ),
    ]

    for question in questions:
        print("\nQUESTION:")
        print(question)

        print("\nTOP RESULTS:")

        for result in rag.retrieve(
            question
        ):
            print(
                f"{result['score']:.4f}",
                "|",
                result["source"],
                "|",
                result["text"],
            )

        print("\nRAG PROMPT:")
        print(
            rag.build_prompt(
                question,
                top_k=2,
            )
        )

    print("\nГлавный вывод:")
    print(
        "RAG не изменяет weights LLM: "
        "он сначала находит нужные Chunks, "
        "а затем добавляет их в Context."
    )


if __name__ == "__main__":
    main()
