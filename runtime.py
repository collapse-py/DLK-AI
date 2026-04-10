import torch
from brain import Brain

class BrainRuntime:
    def __init__(self, vocab_size=50000, d_model=512, device="cpu"):

        self.device = device

        # 固定 brain（架構不變）
        self.model = Brain(vocab_size, d_model).to(device)

        self.current_weights = None
        self.current_path = None

    # =========================
    # load weights
    # =========================
    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state_dict)

        self.current_weights = state_dict
        self.current_path = path

        print(f"[LOAD] {path}")

    # =========================
    # hot swap
    # =========================
    def swap_weights(self, path):
        print(f"[SWAP] {self.current_path} → {path}")
        self.load_weights(path)

    # =========================
    # inference
    # =========================
    def infer(self, x):

        self.model.eval()

        with torch.no_grad():
            return self.model(x)

    # =========================
    # debug info
    # =========================
    def info(self):
        print("Current model:", self.current_path)
        print("Device:", self.device)