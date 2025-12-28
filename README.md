#  BDH Language Model (Post-Transformer AI)

A **character-level language model** built using a **post-Transformer architecture** called **BDH (Beyond Deep Transformers)**.  
This project demonstrates how alternative architectures can model language **without standard self-attention blocks**, focusing on **sparse latent interactions**.

---

##  Project Overview

- Trains a neural language model on the **Tiny Shakespeare dataset**
- Uses a **custom BDH architecture** instead of a traditional Transformer
- Works at **byte / character level**
- Supports **training and interactive text generation**
- Designed to be **hackathon, research, and demo ready**

---

##  Architecture Highlights

- ❌ No standard Transformer blocks
- ✅ Sparse latent projections
- ✅ Rotary positional encoding (RoPE-style)
- ✅ Custom attention via latent space interaction
- ✅ Lightweight & research-oriented design

> This model explores ideas **beyond Transformers**, aligning with modern research directions in efficient and alternative sequence modeling.

---

##  Project Structure

## 📁 Project Structure

```text
BDH/
├── train.py          # Script for training the BDH language model
├── infer.py          # Script for text generation (inference)
├── bdh.py            # Core BDH model architecture
├── input.txt         # Training dataset (Tiny Shakespeare)
├── requirements.txt  # Python dependencies
├── README.md         # Project documentation
```
## Output 
<img width="1436" height="211" alt="image" src="https://github.com/user-attachments/assets/f50913bd-aa65-48e0-acd1-25b3fc589748" />

