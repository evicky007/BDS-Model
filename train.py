# Copyright Pathway Technology, Inc.

import os
from contextlib import nullcontext

import bdh
import numpy as np
import requests
import torch
import torch.nn.functional as F

# ---------------- DEVICE & DTYPE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float32"   # safer for CPU
)

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

ctx = (
    torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    if "cuda" in device.type
    else nullcontext()
)

scaler = torch.amp.GradScaler(
    device=device.type,
    enabled=(dtype == "float16")
)

torch.manual_seed(1337)
print(f"Using device: {device} with dtype {dtype}")

# ---------------- CONFIG ----------------
BDH_CONFIG = bdh.BDHConfig(
    n_layer=3,
    n_embd=128,
    n_head=2,
)

BLOCK_SIZE = 128
BATCH_SIZE = 4
MAX_ITERS = 1000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.1
LOG_FREQ = 50

TEST_PROMPT = "hello good morning "

input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

# ---------------- DATA ----------------
def fetch_data():
    if not os.path.exists(input_file_path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w", encoding="utf-8") as f:
            f.write(requests.get(url).text)

def get_batch(split):
    data = np.memmap(input_file_path, dtype=np.uint8, mode="r")
    if split == "train":
        data = data[: int(0.9 * len(data))]
    else:
        data = data[int(0.9 * len(data)):]

    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

# ---------------- GENERATION HELPER ----------------
def generate_sample(model, prompt_text):
    model.eval()
    prompt = torch.tensor(
        bytearray(prompt_text, "utf-8"),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        out = model.generate(prompt, max_new_tokens=150, top_k=1)

    text = bytes(
        out.squeeze(0).cpu().numpy().astype("uint8")
    ).decode(errors="ignore")

    model.train()
    return text

# ---------------- TRAINING ----------------
if __name__ == "__main__":
    fetch_data()

    model = bdh.BDH(BDH_CONFIG).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    x, y = get_batch("train")

    checkpoints = {
        int(0.25 * MAX_ITERS): "25%",
        int(0.50 * MAX_ITERS): "50%",
        MAX_ITERS - 1: "100%"
    }

    for step in range(MAX_ITERS):
        with ctx:
            logits, loss = model(x, y)

        x, y = get_batch("train")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % LOG_FREQ == 0:
            print(f"Step {step}/{MAX_ITERS} | Loss: {loss.item():.4f}")

        # -------- SAVE & GENERATE AT STAGES --------
        if step in checkpoints:
            label = checkpoints[step]
            weight_path = f"bdh_weights_{label.replace('%','')}.pt"

            torch.save(model.state_dict(), weight_path)
            print(f"\n✅ Saved model at {label} training → {weight_path}")

            print(f"\n📝 Generated text at {label} training:")
            print("-" * 60)
            print(generate_sample(model, TEST_PROMPT))
            print("-" * 60)

    print("\n🎉 Training complete.")
