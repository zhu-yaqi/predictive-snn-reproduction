
import torch
import torch.nn as nn
from spikingjelly.activation_based import (
    neuron, surrogate, functional
)


class LinearLIF(nn.Module):
    """Linear + LIF, working one time step at a time."""

    def __init__(self, in_dim, out_dim, tau=2.0):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=False)

        self.lif = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode="s"
        )

    def forward(self, x):
        return self.lif(self.fc(x))


class PredictiveSuppressionLayer(nn.Module):
    """Predictive Suppression, paper Eq. (1), (2), (4), (5)."""

    def __init__(
        self, in_dim, hidden_dim,
        alpha=10.0, g_min=0.05
    ):
        super().__init__()

        self.encoder = LinearLIF(in_dim, hidden_dim)
        self.predictor = LinearLIF(hidden_dim, in_dim)

        self.alpha = alpha
        self.g_min = g_min

    def forward(self, x):
        # x: [batch, in_dim]

        # 1. Generate binary latent spikes
        z = self.encoder(x)

        # 2. Reconstruct layer input
        x_hat = self.predictor(z)

        # 3. Prediction error
        e = x - x_hat
        m = e.abs().mean(dim=-1, keepdim=True)

        # 4. Error-driven gate
        g = torch.clamp(
            1.0 - torch.exp(-self.alpha * m),
            min=self.g_min,
            max=1.0
        )

        # 5. Communicated activity
        y = g * z

        return {
            "y": y,
            "z": z,
            "x_hat": x_hat,
            "m": m,
            "g": g
        }


class PredictiveSNN(nn.Module):
    """Three hidden layers: 256, 192, 128."""

    def __init__(self, in_dim=2312, num_classes=10):
        super().__init__()

        dims = [in_dim, 256, 192, 128]

        self.layers = nn.ModuleList([
            PredictiveSuppressionLayer(dims[i], dims[i + 1])
            for i in range(3)
        ])

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x_seq):
        # x_seq: [T, B, D]
        T = x_seq.shape[0]

        h = x_seq
        records = []

        for layer in self.layers:
            # Save this layer's actual input for reconstruction
            layer_input = h

            outputs = []

            for t in range(T):
                outputs.append(layer(h[t]))

            record = {
                key: torch.stack(
                    [item[key] for item in outputs], dim=0
                )
                for key in outputs[0].keys()
            }

            record["x"] = layer_input

            records.append(record)
            h = record["y"]

        # Time-averaged classification
        logits = self.classifier(h.mean(dim=0))

        return logits, records


def compute_loss(logits, labels, records):
    """Paper Eq. (6)."""

    task_loss = nn.functional.cross_entropy(
        logits, labels
    )

    pred_losses = []
    firing_rates = []

    for record in records:
        pred_losses.append(
            nn.functional.mse_loss(
                record["x_hat"], record["x"]
            )
        )

        # Predictor output spikes are x_hat in this prototype
        firing_rates.append(record["x_hat"].mean())

    # Layer aggregation is an implementation assumption
    pred_loss = torch.stack(pred_losses).mean()
    pred_rate = torch.stack(firing_rates).mean()

    rate_loss = torch.relu(0.05 - pred_rate).square()

    beta = 0.1
    lam = 2.0

    total_loss = task_loss + beta * (
        pred_loss + lam * rate_loss
    )

    return total_loss, {
        "task": task_loss.detach(),
        "pred": pred_loss.detach(),
        "rate": rate_loss.detach(),
        "pred_rate": pred_rate.detach()
    }