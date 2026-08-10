"""Функции активации — чистый Python-пример для главы 2 проекта Almaz_AI.

Скрипт:
1. показывает работу ReLU, Sigmoid, Tanh и Leaky ReLU;
2. строит их графики;
3. сравнивает функции на одинаковых входных значениях;
4. обучает одинаковую нейросеть «Светофор» с разными активациями;
5. сравнивает Loss и Accuracy;
6. отдельно проверяет модель без скрытой функции активации.

Связанный Notebook:
Labs/chapter02/02_activation_functions.ipynb
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn


RANDOM_SEED = 42
EPOCHS = 500
LEARNING_RATE = 0.05
LEAKY_RELU_SLOPE = 0.01
DECISION_THRESHOLD = 0.5


class TrafficLightNetwork(nn.Module):
    """Нейросеть светофора с заменяемой скрытой функцией активации."""

    def __init__(self, activation: nn.Module | None) -> None:
        """Создаёт два Linear-слоя и, при необходимости, активацию между ними."""

        super().__init__()

        layers: list[nn.Module] = [
            nn.Linear(3, 4),
        ]

        if activation is not None:
            layers.append(activation)

        layers.append(nn.Linear(4, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Выполняет прямой проход и возвращает logits модели."""

        return self.network(features)


def manual_relu(values: torch.Tensor) -> torch.Tensor:
    """Вычисляет ReLU вручную: max(0, x)."""

    return torch.maximum(
        values,
        torch.zeros_like(values),
    )


def manual_sigmoid(values: torch.Tensor) -> torch.Tensor:
    """Вычисляет Sigmoid вручную."""

    return 1 / (1 + torch.exp(-values))


def manual_tanh(values: torch.Tensor) -> torch.Tensor:
    """Вычисляет Tanh через экспоненциальную формулу."""

    exp_positive = torch.exp(values)
    exp_negative = torch.exp(-values)

    return (exp_positive - exp_negative) / (
        exp_positive + exp_negative
    )


def manual_leaky_relu(
    values: torch.Tensor,
    negative_slope: float = LEAKY_RELU_SLOPE,
) -> torch.Tensor:
    """Вычисляет Leaky ReLU вручную."""

    return torch.where(
        values >= 0,
        values,
        negative_slope * values,
    )


def create_activation_table() -> pd.DataFrame:
    """Создаёт таблицу значений четырёх функций активации."""

    values = torch.tensor(
        [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0],
        dtype=torch.float32,
    )

    return pd.DataFrame(
        {
            "x": values.numpy(),
            "ReLU": manual_relu(values).numpy(),
            "Sigmoid": manual_sigmoid(values).numpy(),
            "Tanh": manual_tanh(values).numpy(),
            "Leaky ReLU": manual_leaky_relu(values).numpy(),
        }
    )


def plot_activation(
    x: torch.Tensor,
    y: torch.Tensor,
    title: str,
    ylabel: str,
) -> None:
    """Строит отдельный график функции активации."""

    plt.figure(figsize=(9, 5))
    plt.plot(x.numpy(), y.numpy())
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_all_activations() -> None:
    """Строит отдельные и общий графики функций активации."""

    x = torch.linspace(-6, 6, steps=300)

    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    leaky_relu = nn.LeakyReLU(
        negative_slope=LEAKY_RELU_SLOPE
    )

    relu_y = relu(x)
    sigmoid_y = sigmoid(x)
    tanh_y = tanh(x)
    leaky_y = leaky_relu(x)

    plot_activation(
        x=x,
        y=relu_y,
        title="ReLU",
        ylabel="ReLU(x)",
    )

    plot_activation(
        x=x,
        y=sigmoid_y,
        title="Sigmoid",
        ylabel="Sigmoid(x)",
    )

    plot_activation(
        x=x,
        y=tanh_y,
        title="Tanh",
        ylabel="Tanh(x)",
    )

    plot_activation(
        x=x,
        y=leaky_y,
        title="Leaky ReLU",
        ylabel="LeakyReLU(x)",
    )

    plt.figure(figsize=(10, 6))

    plt.plot(x.numpy(), relu_y.numpy(), label="ReLU")
    plt.plot(x.numpy(), sigmoid_y.numpy(), label="Sigmoid")
    plt.plot(x.numpy(), tanh_y.numpy(), label="Tanh")
    plt.plot(x.numpy(), leaky_y.numpy(), label="Leaky ReLU")

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.title("Сравнение функций активации")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def create_traffic_light_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Создаёт полный датасет состояний учебного светофора."""

    features = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
            [0.0],
            [0.0],
            [0.0],
            [0.0],
        ],
        dtype=torch.float32,
    )

    return features, targets


def calculate_accuracy(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Вычисляет Accuracy бинарной модели."""

    model.eval()

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        predictions = (
            probabilities >= DECISION_THRESHOLD
        ).float()

    correct = predictions == targets

    return float(correct.float().mean().item())


def train_model(
    model: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> tuple[list[float], list[float]]:
    """Обучает модель и возвращает историю Loss и Accuracy."""

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for _ in range(epochs):
        model.train()

        logits = model(features)
        loss = loss_function(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = calculate_accuracy(
            model=model,
            features=features,
            targets=targets,
        )

        loss_history.append(float(loss.item()))
        accuracy_history.append(accuracy)

    return loss_history, accuracy_history


def train_activation_variant(
    activation_factory: Callable[[], nn.Module] | None,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[nn.Module, list[float], list[float]]:
    """Создаёт и обучает одну модель с выбранной активацией."""

    torch.manual_seed(RANDOM_SEED)

    activation = (
        activation_factory()
        if activation_factory is not None
        else None
    )

    model = TrafficLightNetwork(
        activation=activation,
    )

    loss_history, accuracy_history = train_model(
        model=model,
        features=features,
        targets=targets,
    )

    return model, loss_history, accuracy_history


def run_activation_comparison(
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[pd.DataFrame, dict[str, list[float]], dict[str, list[float]]]:
    """Обучает модели с разными активациями и собирает результаты."""

    variants: dict[
        str,
        Callable[[], nn.Module] | None,
    ] = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "Leaky ReLU": lambda: nn.LeakyReLU(
            negative_slope=LEAKY_RELU_SLOPE
        ),
        "Без активации": None,
    }

    rows: list[dict[str, float | str]] = []
    loss_histories: dict[str, list[float]] = {}
    accuracy_histories: dict[str, list[float]] = {}

    for name, activation_factory in variants.items():
        _, loss_history, accuracy_history = train_activation_variant(
            activation_factory=activation_factory,
            features=features,
            targets=targets,
        )

        loss_histories[name] = loss_history
        accuracy_histories[name] = accuracy_history

        rows.append(
            {
                "Activation": name,
                "Final Loss": loss_history[-1],
                "Accuracy": accuracy_history[-1],
            }
        )

    results = pd.DataFrame(rows)

    return results, loss_histories, accuracy_histories


def plot_training_comparison(
    histories: dict[str, list[float]],
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Строит общий график истории обучения для нескольких моделей."""

    plt.figure(figsize=(11, 6))

    for name, values in histories.items():
        plt.plot(values, label=name)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """Запускает все демонстрации главы 2."""

    print("=== ТАБЛИЦА ФУНКЦИЙ АКТИВАЦИИ ===")

    activation_table = create_activation_table()
    print(
        activation_table
        .round(4)
        .to_string(index=False)
    )

    print("\n=== ГРАФИКИ ФУНКЦИЙ ===")
    plot_all_activations()

    features, targets = create_traffic_light_dataset()

    print("\n=== СРАВНЕНИЕ НА НЕЙРОСЕТИ «СВЕТОФОР» ===")

    (
        results,
        loss_histories,
        accuracy_histories,
    ) = run_activation_comparison(
        features=features,
        targets=targets,
    )

    print(
        results
        .round(6)
        .to_string(index=False)
    )

    plot_training_comparison(
        histories=loss_histories,
        title="Сравнение Loss для разных функций активации",
        ylabel="Loss",
    )

    plot_training_comparison(
        histories=accuracy_histories,
        title="Сравнение Accuracy для разных функций активации",
        ylabel="Accuracy",
        ylim=(-0.05, 1.05),
    )

    print("\nГлавный вывод:")
    print(
        "Функция активации меняет способ, которым скрытый слой "
        "преобразует данные, и может заметно влиять на динамику обучения."
    )

    print(
        "Модель без активации также способна решить учебную задачу "
        "светофора, потому что эта конкретная задача линейно разделима."
    )


if __name__ == "__main__":
    main()
