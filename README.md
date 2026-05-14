# Causal Adaptive Streaming Decoder for Imagined Handwriting BCI

This repository contains the implementation of an **Adaptive Confidence-Based Streaming Decoder** for Brain-Computer Interfaces (BCIs) that decode imagined handwriting. This project builds upon the groundbreaking work by **Willett et al. (Nature, 2021)**, reframing handwriting decoding as a real-time streaming inference problem.

## 🚀 Overview

Existing BCI systems for imagined handwriting often rely on fixed-duration decoding windows or fixed output delays. While effective, these methods do not adapt to varying signal clarity. 

Our approach introduces a **causal, adaptive streaming rule** that considers:
1.  **Model Confidence:** The maximum probability assigned to a character.
2.  **Prediction Stability:** The consistency of the predicted class over multiple time steps.

By emitting characters only when both criteria are met, the decoder optimizes the latency-accuracy trade-off in real-time.

## 🧠 Model Architecture

The core of the system is a **Two-Layer GRU (Gated Recurrent Unit)**:
- **Input:** 192-channel neural signals (multi-unit threshold-crossing rates binned at 20ms).
- **Hidden Units:** 256 per layer.
- **Output:** 31 classes (26 lowercase letters + special characters).
- **Inference:** Causal processing with a rolling buffer.

## 📊 Key Results

Our adaptive decoder demonstrates significant improvements over baseline methods:

| Method | CER (%) | Latency (ms/char) |
| :--- | :--- | :--- |
| **Adaptive Decoder (Best)** | **27.7%** | **564 ms** |
| Fixed-Delay Baseline | 26.9% | 600 ms |
| Fixed-Window Baseline | 100.7% | 3000 ms |

*The results show that adaptive waiting is more efficient than fixed delays, as it avoids unnecessary waiting when neural signals are clear.*

## 📁 Repository Structure

- `Copy_of_Final_DL_Project_(1).ipynb`: Full implementation, training pipeline, and evaluation logic.
- `DL_Project_Report.pdf`: Detailed technical report explaining the methodology, literature review, and analysis.
- `DL_presentation_slide.pdf`: Presentation deck summarizing the project findings.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch / TensorFlow (implemented in Jupyter)
- **Environment:** Google Colab (T4 GPU)
- **Dataset:** Pre-aligned neural recordings from the [Willett et al. (2021) Nature dataset](https://www.nature.com/articles/s41586-021-03506-2).

## 🔗 References

- Willett, F. R., et al. (2021). "High-performance brain-to-text communication via handwriting." *Nature*, 593(7858), 249-254.
