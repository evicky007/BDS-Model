import torch
import bdh

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same config used during training
config = bdh.BDHConfig(
    n_layer=3,
    n_embd=128,
    n_head=2,
)

# Prompt input
prompt_text = input("Enter prompt: ")

prompt = torch.tensor(
    bytearray(prompt_text, "utf-8"),
    dtype=torch.long,
    device=device
).unsqueeze(0)

# Helper function for loading model & generating text
def generate_from_checkpoint(weight_path, label):
    print("\n" + "=" * 60)
    print(f"🔹 Output at {label} Training")
    print("=" * 60)

    model = bdh.BDH(config).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=200, top_k=1)

    generated_text = bytes(
        output.squeeze(0).cpu().numpy().astype("uint8")
    ).decode(errors="ignore")

    print(generated_text)
    print("=" * 60)


# Generate outputs at different training stages
generate_from_checkpoint("bdh_weights_25.pt", "25%")
generate_from_checkpoint("bdh_weights_50.pt", "50%")
generate_from_checkpoint("bdh_weights_100.pt", "100%")
