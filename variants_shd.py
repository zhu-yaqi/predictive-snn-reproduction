
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, surrogate

from model import LinearLIF


class HardGateSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, theta):
        ctx.save_for_backward(gate)
        return (gate > theta).to(gate.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # Identity straight-through estimator
        return grad_output, None


class VariantLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        variant,
        alpha=10.0,
        g_min=0.05,
        theta=0.3,
        gamma=4.0
    ):
        super().__init__()

        assert variant in ("error", "hard_ste", "respike")

        self.variant = variant
        self.alpha = alpha
        self.g_min = g_min
        self.theta = theta
        self.gamma = gamma

        # Same encoder/predictor skeleton
        self.encoder = LinearLIF(in_dim, hidden_dim)
        self.predictor = LinearLIF(hidden_dim, in_dim)

        if variant == "error":
            # Positive and negative errors each have in_dim features.
            # Output dimension = 2 * in_dim.
            self.error_lif = neuron.LIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode="s"
            )

        elif variant == "respike":
            self.respike_lif = neuron.LIFNode(
                tau=2.0,
                surrogate_function=surrogate.ATan(),
                detach_reset=True,
                step_mode="s"
            )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.predictor(z)

        e = x - x_hat
        m = e.abs().mean(dim=-1, keepdim=True)

        g = torch.clamp(
            1.0 - torch.exp(-self.alpha * m),
            min=self.g_min,
            max=1.0
        )

        if self.variant == "error":
            signed_error = torch.cat(
                [torch.relu(e), torch.relu(-e)],
                dim=-1
            )
            y = self.error_lif(signed_error)

        elif self.variant == "hard_ste":
            hard_gate = HardGateSTE.apply(g, self.theta)
            y = hard_gate * z

        elif self.variant == "respike":
            y = self.respike_lif(
                self.gamma * g * z
            )

        return {
            "y": y,
            "z": z,
            "x_hat": x_hat,
            "m": m,
            "g": g
        }


class VariantSNN(nn.Module):
    def __init__(
        self,
        variant,
        in_dim=700,
        num_classes=20
    ):
        super().__init__()

        assert variant in ("error", "hard_ste", "respike")

        self.variant = variant
        self.layers = nn.ModuleList()

        current_dim = in_dim

        for hidden_dim in [256, 192, 128]:
            layer = VariantLayer(
                in_dim=current_dim,
                hidden_dim=hidden_dim,
                variant=variant,
                theta=0.1
            )
            self.layers.append(layer)

            if variant == "error":
                current_dim = current_dim * 2
            else:
                current_dim = hidden_dim

        # Error variant outputs residual spikes with a different width.
        self.classifier = nn.Linear(
            current_dim, num_classes
        )

    def forward(self, x_seq):
        h = x_seq
        records = []

        for layer in self.layers:
            layer_input = h
            outputs = []

            for t in range(h.shape[0]):
                outputs.append(layer(h[t]))

            record = {
                key: torch.stack(
                    [item[key] for item in outputs],
                    dim=0
                )
                for key in outputs[0]
            }

            record["x"] = layer_input

            records.append(record)
            h = record["y"]

        logits = self.classifier(h.mean(dim=0))

        return logits, records