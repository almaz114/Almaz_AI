"""Глава 10: AI-агенты и Tools.

Чистый Python-пример проекта Almaz_AI.

Скрипт показывает:
- Python Tools;
- Tool Registry;
- Dispatcher;
- структурированные Tool Calls;
- Observation;
- простой многошаговый Agent Loop;
- ограничение MAX_STEPS;
- базовое разделение read-only / write Tools.
"""

from __future__ import annotations

from typing import Any, Callable


MAX_STEPS = 5


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
    """Считает слова в тексте."""

    return {
        "status": "ok",
        "result": len(text.split()),
    }


NOTES = {
    "intro": (
        "Искусственный интеллект использует "
        "модели и данные."
    ),
    "agent": (
        "Агент может выбирать инструменты "
        "и использовать их результаты."
    ),
}


def get_note(
    name: str,
) -> dict[str, Any]:
    """Возвращает учебную заметку."""

    if name not in NOTES:
        return {
            "status": "error",
            "message": (
                f"Заметка '{name}' не найдена."
            ),
        }

    return {
        "status": "ok",
        "result": NOTES[name],
    }


TOOL_REGISTRY: dict[
    str,
    Callable[..., dict[str, Any]],
] = {
    "add_numbers": add_numbers,
    "word_count": word_count,
    "get_note": get_note,
}


TOOL_SCHEMAS = [
    {
        "name": "add_numbers",
        "description": (
            "Возвращает сумму двух чисел a и b."
        ),
        "parameters": {
            "a": "number",
            "b": "number",
        },
    },
    {
        "name": "word_count",
        "description": (
            "Возвращает количество слов "
            "в строке text."
        ),
        "parameters": {
            "text": "string",
        },
    },
    {
        "name": "get_note",
        "description": (
            "Возвращает текст заметки "
            "по имени name."
        ),
        "parameters": {
            "name": "string",
        },
    },
]


READ_ONLY_TOOLS = {
    "add_numbers",
    "word_count",
    "get_note",
}


def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Находит Tool по имени и безопасно выполняет."""

    if tool_name not in TOOL_REGISTRY:
        return {
            "status": "error",
            "message": (
                f"Неизвестный Tool: {tool_name}"
            ),
        }

    tool = TOOL_REGISTRY[tool_name]

    try:
        return tool(**arguments)

    except TypeError as error:
        return {
            "status": "error",
            "message": (
                f"Ошибка аргументов: {error}"
            ),
        }

    except Exception as error:
        return {
            "status": "error",
            "message": (
                f"Ошибка Tool: {error}"
            ),
        }


def simulated_model(
    user_message: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Симулирует структурированный ответ LLM.

    В настоящем агенте эту функцию можно заменить
    вызовом локальной Qwen с поддержкой Tool Calling.
    """

    message = user_message.lower()

    if "сколько слов" in message:
        if not observations:
            return {
                "type": "tool_call",
                "tool_name": "get_note",
                "arguments": {
                    "name": "agent",
                },
            }

        last = observations[-1]

        if (
            last["tool_name"] == "get_note"
            and last["result"]["status"] == "ok"
        ):
            return {
                "type": "tool_call",
                "tool_name": "word_count",
                "arguments": {
                    "text": (
                        last["result"]["result"]
                    ),
                },
            }

        if (
            last["tool_name"] == "word_count"
            and last["result"]["status"] == "ok"
        ):
            return {
                "type": "final",
                "content": (
                    "В заметке "
                    f"{last['result']['result']} слов."
                ),
            }

    return {
        "type": "final",
        "content": (
            "Для этой учебной задачи "
            "дополнительный Tool не нужен."
        ),
    }


def run_agent(
    user_message: str,
    max_steps: int = MAX_STEPS,
) -> str:
    """Запускает простой Agent Loop."""

    observations: list[dict[str, Any]] = []

    for step in range(max_steps):
        model_output = simulated_model(
            user_message,
            observations,
        )

        print(
            f"STEP {step + 1}:",
            model_output,
        )

        if model_output["type"] == "final":
            return model_output["content"]

        if model_output["type"] != "tool_call":
            return (
                "Ошибка: неизвестный тип "
                "ответа модели."
            )

        tool_name = model_output["tool_name"]

        if tool_name not in READ_ONLY_TOOLS:
            return (
                f"Tool '{tool_name}' "
                "не разрешён."
            )

        tool_result = execute_tool(
            tool_name,
            model_output["arguments"],
        )

        observation = {
            "tool_name": tool_name,
            "result": tool_result,
        }

        observations.append(observation)

        print(
            "OBSERVATION:",
            observation,
        )

    return (
        "Agent остановлен: "
        "достигнут max_steps."
    )


def main() -> None:
    """Запускает демонстрации главы."""

    print("=== REGISTERED TOOLS ===")

    for schema in TOOL_SCHEMAS:
        print(
            schema["name"],
            "→",
            schema["description"],
        )

    print("\n=== DIRECT TOOL CALL ===")

    result = execute_tool(
        "add_numbers",
        {
            "a": 10,
            "b": 20,
        },
    )

    print(result)

    print("\n=== AGENT LOOP ===")

    answer = run_agent(
        "Сколько слов в заметке agent?"
    )

    print("\nFINAL ANSWER:")
    print(answer)

    print("\nГлавный вывод:")
    print(
        "LLM выбирает действие, "
        "а контролируемый Python-код "
        "реально выполняет Tool."
    )


if __name__ == "__main__":
    main()
