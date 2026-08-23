"""Глава 9: Локальные LLM и Qwen.

Чистый Python-пример проекта Almaz_AI.

Требования для практического запуска:
- локально установленный Ollama;
- установленный Python-пакет `ollama`;
- уже загруженная локальная Qwen-модель.

Имя модели задаётся через MODEL_NAME.
"""

from __future__ import annotations

import time
from typing import Any


MODEL_NAME = "your-local-qwen-model"
DEFAULT_TEMPERATURE = 0.2


def get_ollama_module() -> Any:
    """Импортирует ollama и выдаёт понятную ошибку при отсутствии пакета."""

    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "Python-пакет 'ollama' не установлен."
        ) from exc

    return ollama


def ask_local_model(
    prompt: str,
    *,
    model_name: str = MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    system_prompt: str | None = None,
) -> str:
    """Отправляет одиночный запрос локальной модели."""

    ollama = get_ollama_module()

    messages: list[dict[str, str]] = []

    if system_prompt is not None:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": temperature,
        },
    )

    return response["message"]["content"]


def timed_request(
    prompt: str,
    *,
    model_name: str = MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[str, float]:
    """Выполняет запрос и возвращает ответ и общее время."""

    started = time.perf_counter()

    answer = ask_local_model(
        prompt,
        model_name=model_name,
        temperature=temperature,
    )

    elapsed = time.perf_counter() - started

    return answer, elapsed


def run_multi_turn_chat(
    *,
    model_name: str = MODEL_NAME,
) -> None:
    """Показывает простую историю диалога."""

    ollama = get_ollama_module()

    messages = [
        {
            "role": "system",
            "content": (
                "Ты преподаватель PyTorch. "
                "Отвечай кратко и технически корректно."
            ),
        },
        {
            "role": "user",
            "content": "Что такое Tensor?",
        },
    ]

    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": DEFAULT_TEMPERATURE,
        },
    )

    first_answer = response["message"]["content"]

    print("Assistant:")
    print(first_answer)

    messages.append(
        {
            "role": "assistant",
            "content": first_answer,
        }
    )

    messages.append(
        {
            "role": "user",
            "content": (
                "Чем Tensor отличается "
                "от списка Python?"
            ),
        }
    )

    response = ollama.chat(
        model=model_name,
        messages=messages,
        options={
            "temperature": DEFAULT_TEMPERATURE,
        },
    )

    print("\nAssistant:")
    print(response["message"]["content"])


def compare_temperatures(
    prompt: str,
    *,
    model_name: str = MODEL_NAME,
) -> None:
    """Сравнивает два режима Sampling."""

    for temperature in (0.1, 1.0):
        print(
            f"\n=== temperature={temperature} ==="
        )

        answer = ask_local_model(
            prompt,
            model_name=model_name,
            temperature=temperature,
        )

        print(answer)


def main() -> None:
    """Запускает демонстрации локальной LLM."""

    if MODEL_NAME == "your-local-qwen-model":
        print(
            "Сначала укажи в MODEL_NAME имя "
            "своей установленной локальной Qwen-модели."
        )
        return

    try:
        answer, elapsed = timed_request(
            "Объясни в трёх предложениях, что такое Token."
        )
    except RuntimeError as error:
        print(error)
        return
    except Exception as error:
        print(
            "Не удалось выполнить локальный запрос:",
            error,
        )
        return

    print("=== FIRST REQUEST ===")
    print(answer)
    print(f"\nВремя: {elapsed:.2f} сек.")

    compare_temperatures(
        "Придумай короткую аналогию для механизма Attention."
    )

    print("\n=== MULTI-TURN CHAT ===")
    run_multi_turn_chat()

    print("\nГлавный вывод:")
    print(
        "Prompting и Sampling меняют использование модели, "
        "но не изменяют её обученные weights."
    )


if __name__ == "__main__":
    main()
