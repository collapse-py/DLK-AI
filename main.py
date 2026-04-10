import torch
from runtime import BrainRuntime

# =========================
# init runtime
# =========================
rt = BrainRuntime(
    vocab_size=50000,
    d_model=512,
    device="cpu"
)

# =========================
# load model A
# =========================
rt.load_weights(r"F:\ai\brain_weights.pt")

x = torch.randint(0, 1000, (1, 16))

out1 = rt.infer(x)
print("A output:", out1.shape)

# =========================
# swap model B
# =========================
rt.swap_weights(r"F:\ai\brain_weights_b.pt")

out2 = rt.infer(x)
print("B output:", out2.shape)