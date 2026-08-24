"""Глава 13: итоговый локальный AI-ассистент.

Учебный проект Almaz_AI.

Объединяет:
- простой локальный RAG;
- Tool Registry;
- Dispatcher;
- Agent Controller;
- MAX_STEPS;
- место для подключения локальной Qwen через Ollama.

Модельное решение в демонстрации симулируется, чтобы
архитектура работала без внешних зависимостей.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable

import torch


MAX_STEPS = 5
TOP_K = 2
MODEL_NAME = "your-local-qwen-model"


DOCUMENTS = [
    {
        "source": "ai_course.md",
        "text": (
            "RAG добавляет найденные фрагменты документов "
            "в контекст модели и не изменяет её weights."
        ),
    },
    {
        "source": "lora.md",
        "text": (
            "LoRA замораживает базовые веса модели "
            "и обучает небольшие low-rank адаптеры."
        ),
    },
    {
        "source": "agents.md",
        "text": (
            "AI-агент может выбирать инструменты, "
            "получать Observation и продолжать работу."
        ),
    },
]


def tokenize(text: str) -> list[str]:
    """Простейшая токенизация."""

    return re.findall(
        r"[а-яёa-z0-9]+",
        text.lower(),
    )


class SimpleKnowledgeBase:
    """Минимальный локальный RAG."""

    def __init__(
        self,
        documents: list[dict[str, str]],
    ) -> None:
        self.documents = documents

        self.vocabulary = sorted(
            {
                token
                for document in documents
                for token in tokenize(document["text"])
            }
        )

        self.token_to_id = {
            token: index
            for index, token in enumerate(self.vocabulary)
        }

        self.document_frequency: Counter[str] = Counter()

        for document in documents:
            for token in set(tokenize(document["text"])):
                self.document_frequency[token] += 1

        self.document_vectors = torch.stack(
            [
                self.text_to_vector(document["text"])
                for document in documents
            ]
        )

    def text_to_vector(
        self,
        text: str,
    ) -> torch.Tensor:
        """Создаёт TF-IDF-подобный vector."""

        tokens = tokenize(text)
        counts = Counter(tokens)

        vector = torch.zeros(
            len(self.vocabulary),
            dtype=torch.float32,
        )

        for token, count in counts.items():
            token_id = self.token_to_id.get(token)

            if token_id is None:
                continue

            tf = count / max(len(tokens), 1)
            df = self.document_frequency.get(token, 0)

            idf = math.log(
                (len(self.documents) + 1)
                / (df + 1)
            ) + 1.0

            vector[token_id] = tf * idf

        return vector

    @staticmethod
    def cosine_similarity(
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> float:
        """Считает Cosine Similarity."""

        denominator = (
            torch.linalg.vector_norm(a)
            * torch.linalg.vector_norm(b)
        )

        if denominator.item() == 0:
            return 0.0

        return float(
            torch.dot(a, b) / denominator
        )

    def search(
        self,
        question: str,
        *,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        """Возвращает наиболее похожие документы."""

        query_vector = self.text_to_vector(question)
        results: list[dict[str, Any]] = []

        for document, vector in zip(
            self.documents,
            self.document_vectors,
        ):
            score = self.cosine_similarity(
                query_vector,
                vector,
            )

            results.append(
                {
                    **document,
                    "score": score,
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]


KNOWLEDGE_BASE = SimpleKnowledgeBase(DOCUMENTS)


def add_numbers(
    a: float,
    b: float,
) -> dict[str, Any]:
    """Складывает два числа."""

    return {
        "status": "ok",
        "result": a + b,
    }


def word_count(
    text: str,
) -> dict[str, Any]:
    """Считает количество слов."""

    return {
        "status": "ok",
        "result": len(text.split()),
    }


def search_knowledge_base(
    question: str,
) -> dict[str, Any]:
    """Read-only Tool поиска знаний."""

    return {
        "status": "ok",
        "results": KNOWLEDGE_BASE.search(question),
    }


TOOL_REGISTRY: dict[
    str,
    Callable[..., dict[str, Any]],
] = {
    "add_numbers": add_numbers,
    "word_count": word_count,
    "search_knowledge_base": search_knowledge_base,
}


READ_ONLY_TOOLS = set(TOOL_REGISTRY)


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Выполняет разрешённый Tool."""

    if tool_name not in READ_ONLY_TOOLS:
        return {
            "status": "error",
            "message": f"Tool '{tool_name}' не разрешён.",
        }

    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return {
            "status": "error",
            "message": f"Tool '{tool_name}' не найден.",
        }

    try:
        return tool(**arguments)

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }


def simulated_llm_decision(
    user_message: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Симулирует решение LLM для учебного Agent Loop."""

    text = user_message.lower()

    if "сложи" in text:
        numbers = re.findall(
            r"-?\d+(?:\.\d+)?",
            text,
        )

        if len(numbers) >= 2:
            return {
                "type": "tool",
                "name": "add_numbers",
                "arguments": {
                    "a": float(numbers[0]),
                    "b": float(numbers[1]),
                },
            }

    if "сколько слов" in text:
        if not observations:
            return {
                "type": "rag",
                "query": user_message,
            }

        last = observations[-1]

        if last["kind"] == "rag":
            results = last["result"]["results"]

            if not results:
                return {
                    "type": "final",
                    "content": "В базе знаний ничего не найдено.",
                }

            return {
                "type": "tool",
                "name": "word_count",
                "arguments": {
                    "text": results[0]["text"],
                },
            }

        if (
            last["kind"] == "tool"
            and last["name"] == "word_count"
            and last["result"]["status"] == "ok"
        ):
            return {
                "type": "final",
                "content": (
                    "В найденном фрагменте "
                    f"{last['result']['result']} слов."
                ),
            }

    if any(
        keyword in text
        for keyword in ("rag", "lora", "агент", "документ")
    ):
        if not observations:
            return {
                "type": "rag",
                "query": user_message,
            }

        last = observations[-1]

        if last["kind"] == "rag":
            results = last["result"]["results"]

            if not results:
                return {
                    "type": "final",
                    "content": "В базе знаний ничего не найдено.",
                }

            best = results[0]

            return {
                "type": "final",
                "content": (
                    f"По базе знаний: {best['text']} "
                    f"(источник: {best['source']})"
                ),
            }

    return {
        "type": "final",
        "content": "Это прямой ответ модели без Tool и RAG.",
    }


class AssistantController:
    """Управляет Agent Loop."""

    def __init__(
        self,
        *,
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.max_steps = max_steps

    def run(
        self,
        user_message: str,
    ) -> str:
        """Выполняет один запрос пользователя."""

        observations: list[dict[str, Any]] = []

        for step in range(self.max_steps):
            decision = simulated_llm_decision(
                user_message,
                observations,
            )

            print(
                f"STEP {step + 1}:",
                decision,
            )

            if decision["type"] == "final":
                return decision["content"]

            if decision["type"] == "rag":
                result = search_knowledge_base(
                    decision["query"]
                )

                observations.append(
                    {
                        "kind": "rag",
                        "result": result,
                    }
                )

                continue

            if decision["type"] == "tool":
                result = execute_tool(
                    decision["name"],
                    decision["arguments"],
                )

                observations.append(
                    {
                        "kind": "tool",
                        "name": decision["name"],
                        "result": result,
                    }
                )

                continue

            return (
                "Ошибка Controller: "
                "неизвестный тип решения."
            )

        return (
            "Assistant остановлен: "
            "достигнут MAX_STEPS."
        )


def ask_qwen(
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    """Пример подключения локальной Qwen через Ollama."""

    try:
        import ollama
    except ImportError:
        return {
            "status": "error",
            "message": "Python-пакет ollama не установлен.",
        }

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": 0.2,
            },
        )

        return {
            "status": "ok",
            "content": response["message"]["content"],
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }


def main() -> None:
    """Запускает финальную демонстрацию курса."""

    assistant = AssistantController()

    examples = [
        "Объясни, что такое Tensor.",
        "Сложи 125 и 37.",
        "Как работает LoRA?",
        "Сколько слов в найденной заметке про LoRA?",
    ]

    for user_message in examples:
        print("\n" + "=" * 70)
        print("USER:")
        print(user_message)

        print("\nASSISTANT:")
        print(
            assistant.run(user_message)
        )

    print("\nФинальная архитектура:")
    print(
        "LLM + RAG + Tools + Controller "
        "+ optional LoRA = AI Assistant"
    )


if __name__ == "__main__":
    main()
