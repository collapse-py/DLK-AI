import os
import torch
from safetensors.torch import load_file, save_file

# -----------------------------
# 配置
# setup
# -----------------------------
model_path = r"F:\ai/model.safetensors"
split_dir = r"F:\ai\split"
weights_dir = r"F:\ai\weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create folder
os.makedirs(split_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)

# -----------------------------
# 拆檔 safetensors
# Split gears safetensors
# -----------------------------
print("Splitting safetensors into smaller files...")
state_dict = load_file(model_path, device="cpu")
items = list(state_dict.items())

max_tensors_per_file = 50
file_idx = 0
for i in range(0, len(items), max_tensors_per_file):
    shard_items = dict(items[i:i+max_tensors_per_file])
    shard_path = os.path.join(split_dir, f"shard_{file_idx:03d}.safetensors")
    save_file(shard_items, shard_path)
    print(f"Saved {shard_path} ({len(shard_items)} tensors)")
    file_idx += 1
print("✅ Split complete!")

# -----------------------------
# save file
# -----------------------------
for file_name in sorted(os.listdir(split_dir)):
    if not file_name.endswith(".safetensors"):
        continue

    path = os.path.join(split_dir, file_name)
    print(f"Processing {file_name} ...")
    state_dict = load_file(path, device="cpu")
    state_dict = {k: v.to(device) for k, v in state_dict.items()}

    embedding_keys = [k for k in state_dict if "embedding" in k]
    layer_keys = [k for k in state_dict if "layers" in k]
    output_keys = [k for k in state_dict if "output" in k]

    # Embedding shard
    if embedding_keys:
        embedding_state = {k.split("embedding.")[1]: state_dict[k] for k in embedding_keys}
        torch.save(embedding_state, os.path.join(weights_dir, "embedding.pt"))
        print(f"Saved embedding.pt ({len(embedding_state)} tensors)")

    # Layers shard
    if layer_keys:
        layer_indices = sorted(list(set(int(k.split('.')[2]) for k in layer_keys)))
        for i in layer_indices:
            shard_keys = [k for k in layer_keys if f"layers.{i}" in k]
            layer_state = {k.split(f"layers.{i}.")[1]: state_dict[k] for k in shard_keys}
            torch.save(layer_state, os.path.join(weights_dir, f"layer_{i}.pt"))
            print(f"Saved layer_{i}.pt ({len(layer_state)} tensors)")

    # Output shard
    if output_keys:
        output_state = {k.split("output.")[1]: state_dict[k] for k in output_keys}
        torch.save(output_state, os.path.join(weights_dir, "output.pt"))
        print(f"Saved output.pt ({len(output_state)} tensors)")

print("done")