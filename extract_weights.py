import torch
from safetensors.torch import load_file

# =========================
# input / output
# =========================

model_path = r"F:\ai\model.safetensors"
output_path = r"F:\ai\model_weights.pt"

print("Loading safetensors...")

state_dict = load_file(model_path, device="cpu")

print("Tensors:", len(state_dict))

clean_state = {
    k: v.detach().cpu()
    for k, v in state_dict.items()
}

print("Saving single PyTorch weight file...")

torch.save(clean_state, output_path)

print("DONE →", output_path)
