import torch
import torch.nn as nn


class MetaBrain(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        vocab_size=50000
    ):
        super().__init__()

        self.config = {
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "vocab_size": vocab_size
        }

        # --- Embedding ---
        self.embedding = nn.Embedding(
            vocab_size,
            d_model,
            device="meta"
        )

        # --- Transformer Layers ---
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                device="meta"
            )
            for _ in range(num_layers)
        ])

        # --- Output ---
        self.output_projection = nn.Linear(
            d_model,
            vocab_size,
            device="meta"
        )


def save_brain_skeleton(path="brain_skeleton.pt"):

    brain = MetaBrain()

    torch.save(brain, path)

    print(f"Brain skeleton saved to: {path}")

    print(
        "Parameter count:",
        sum(p.numel() for p in brain.parameters())
    )

    print(
        "CUDA memory allocated:",
        torch.cuda.memory_allocated()
    )


if __name__ == "__main__":

    save_brain_skeleton()