
import torch
from spikingjelly.activation_based import functional

from model import PredictiveSNN, compute_loss


def main():
    torch.manual_seed(42)

    model = PredictiveSNN(
        in_dim=2312,
        num_classes=10
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # Synthetic test data
    T = 16
    B = 4
    D = 2312

    x = torch.rand(T, B, D)
    labels = torch.randint(0, 10, (B,))

    functional.reset_net(model)
    model.train()

    optimizer.zero_grad()

    logits, records = model(x)

    loss, details = compute_loss(
        logits, labels, records
    )

    loss.backward()

    grad = model.layers[0].encoder.fc.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()

    optimizer.step()

    print("Logits:", logits.shape)
    print("Loss:", loss.item())

    for i, record in enumerate(records):
        z = record["z"]
        y = record["y"]
        g = record["g"]

        print(f"\nLayer {i + 1}")
        print("z shape:", z.shape)
        print("y shape:", y.shape)
        print("gate shape:", g.shape)
        print("gate range:", g.min().item(), g.max().item())

        # Binary representation check
        assert torch.all((z == 0) | (z == 1))

        # Communication must not exceed representation
        assert torch.all(y <= z + 1e-6)

        # Gate bounds
        assert torch.all(g >= 0.05 - 1e-6)
        assert torch.all(g <= 1.0 + 1e-6)

    print("\nSmoke test passed!")


if __name__ == "__main__":
    main()