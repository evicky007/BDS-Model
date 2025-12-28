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

## 📁 Project Structure

BDH/
├── train.py # Model training script
├── infer.py # Inference (text generation)
├── bdh.py # BDH model architecture
├── input.txt # Training data (Tiny Shakespeare)
├── requirements.txt # Dependencies
├── README.md # Project documentation


<img width="1004" height="290" alt="image" src="https://github.com/user-attachments/assets/e14b5bcf-4025-4014-8655-1d693f121bf4" />

