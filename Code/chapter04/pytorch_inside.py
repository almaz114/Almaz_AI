"""Глава 4: PyTorch изнутри.

Чистый Python-пример проекта Almaz_AI.

Скрипт демонстрирует:
- Tensor, shape, dtype и device;
- nn.Module;
- nn.Linear;
- forward();
- nn.Sequential;
- parameters() и named_parameters();
- requires_grad и parameter.grad;
- autograd;
- train(), eval(), torch.no_grad();
- подсчёт параметров;
- Shapes внутри модели.
"""

import torch
import torch.nn as nn


RANDOM_SEED = 42


class TrafficLightNetwork(nn.Module):
    """Простая нейросеть для учебной задачи светофора."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Выполняет Forward Pass."""
        hidden = self.layer1(features)
        activated = self.relu(hidden)
        logits = self.layer2(activated)
        return logits


def demonstrate_tensor() -> None:
    """Показывает основные свойства Tensor."""
    tensor = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    print("=== TENSOR ===")
    print(tensor)
    print("shape:", tensor.shape)
    print("dtype:", tensor.dtype)
    print("device:", tensor.device)


def demonstrate_linear() -> None:
    """Показывает внутренности nn.Linear."""
    torch.manual_seed(RANDOM_SEED)
    layer = nn.Linear(3, 4)

    print("\n=== LINEAR(3, 4) ===")
    print(layer)
    print("weight shape:", layer.weight.shape)
    print("bias shape:", layer.bias.shape)
    print("weights:", layer.weight.numel())
    print("bias:", layer.bias.numel())
    print("total parameters:", layer.weight.numel() + layer.bias.numel())


def print_model_parameters(model: nn.Module) -> None:
    """Печатает имена, Shapes и requires_grad параметров."""
    print("\n=== MODEL PARAMETERS ===")

    for name, parameter in model.named_parameters():
        print(
            name,
            "| shape =",
            tuple(parameter.shape),
            "| requires_grad =",
            parameter.requires_grad,
        )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Возвращает общее и обучаемое количество параметров."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def inspect_forward_pass(
    model: TrafficLightNetwork,
    features: torch.Tensor,
) -> None:
    """Показывает Tensor и Shape после каждого слоя."""
    print("\n=== FORWARD PASS ПО СЛОЯМ ===")

    print("Input:")
    print(features)
    print("shape:", tuple(features.shape))

    hidden = model.layer1(features)
    print("\nAfter Linear(3,4):")
    print(hidden)
    print("shape:", tuple(hidden.shape))

    activated = model.relu(hidden)
    print("\nAfter ReLU:")
    print(activated)
    print("shape:", tuple(activated.shape))

    logits = model.layer2(activated)
    print("\nAfter Linear(4,1):")
    print(logits)
    print("shape:", tuple(logits.shape))

    normal_output = model(features)

    print(
        "\nРучной проход совпадает с model(x):",
        torch.allclose(logits, normal_output),
    )


def demonstrate_gradients(
    model: TrafficLightNetwork,
    features: torch.Tensor,
) -> None:
    """Показывает gradients после backward()."""
    target = torch.tensor([[1.0]], dtype=torch.float32)
    loss_function = nn.BCEWithLogitsLoss()

    logits = model(features)
    loss = loss_function(logits, target)

    model.zero_grad()
    loss.backward()

    print("\n=== GRADIENTS ===")
    print("Loss:", loss.item())

    for name, parameter in model.named_parameters():
        print(
            name,
            "| grad shape =",
            tuple(parameter.grad.shape),
        )


def demonstrate_autograd() -> None:
    """Показывает autograd на функции y = x²."""
    value = torch.tensor(2.0, requires_grad=True)
    result = value ** 2
    result.backward()

    print("\n=== AUTOGRAD ===")
    print("x:", value.item())
    print("y=x²:", result.item())
    print("dy/dx:", value.grad.item())


def demonstrate_train_eval(
    model: nn.Module,
    features: torch.Tensor,
) -> None:
    """Показывает режимы train/eval и no_grad."""
    print("\n=== TRAIN / EVAL ===")

    model.train()
    print("После train():", model.training)

    model.eval()
    print("После eval():", model.training)

    with torch.no_grad():
        output = model(features)

    print("Inference output:", output)
    print("output.requires_grad:", output.requires_grad)


def print_state_dict(model: nn.Module) -> None:
    """Печатает содержимое state_dict модели."""
    print("\n=== STATE DICT ===")

    for name, tensor in model.state_dict().items():
        print(name, tuple(tensor.shape))


def demonstrate_batch_shapes(
    model: TrafficLightNetwork,
) -> None:
    """Показывает изменение Shape для Batch."""
    batch = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ], dtype=torch.float32)

    print("\n=== BATCH SHAPES ===")
    print("Input:", tuple(batch.shape))

    hidden = model.layer1(batch)
    print("After Linear 1:", tuple(hidden.shape))

    activated = model.relu(hidden)
    print("After ReLU:", tuple(activated.shape))

    output = model.layer2(activated)
    print("After Linear 2:", tuple(output.shape))


def main() -> None:
    """Запускает демонстрации главы 4."""
    torch.manual_seed(RANDOM_SEED)

    demonstrate_tensor()
    demonstrate_linear()

    model = TrafficLightNetwork()

    print("\n=== MODEL ===")
    print(model)

    print_model_parameters(model)

    total, trainable = count_parameters(model)
    print("\nВсего параметров:", total)
    print("Обучаемых параметров:", trainable)

    example = torch.tensor(
        [[0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )

    inspect_forward_pass(model, example)
    demonstrate_gradients(model, example)
    demonstrate_autograd()
    demonstrate_train_eval(model, example)
    print_state_dict(model)
    demonstrate_batch_shapes(model)

    print("\nГлавный вывод:")
    print(
        "PyTorch-модель — это nn.Module, который получает Tensor, "
        "последовательно применяет слои и хранит обучаемые параметры."
    )


if __name__ == "__main__":
    main()
