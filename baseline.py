
import torch
import torch.nn as nn

from spikingjelly.activation_based import (
    neuron, surrogate
)


class BaselineSNN(nn.Module):

    def __init__(
        self,
        in_dim=2312,
        num_classes=10,
        tau=2.0
    ):
        super().__init__()

        dims = [in_dim, 256, 192, 128]

        self.layers = nn.ModuleList()

        for i in range(3):
            block = nn.Sequential(
                nn.Linear(
                    dims[i],
                    dims[i + 1],
                    bias=False
                ),
                neuron.LIFNode(
                    tau=tau,
                    surrogate_function=surrogate.ATan(),
                    detach_reset=True,
                    step_mode="s"
                )
            )
            self.layers.append(block)

        self.classifier = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x_seq):
        """
        Input:
            [T, B, 2312]

        Output:
            logits: [B, 10]
            records: layer activities
        """
        h = x_seq
        records = []

        for layer in self.layers:
            outputs = []

            for t in range(h.shape[0]):
                outputs.append(layer(h[t]))

            h = torch.stack(outputs, dim=0)

            # Baseline forwards all representation spikes.
            records.append({
                "z": h,
                "y": h
            })

        logits = self.classifier(
            h.mean(dim=0)
        )

        return logits, records