# ShaunGPT — Understanding Intelligence, One Token at a Time. 🤖

### A GPT-Style Language Model Built from Scratch Using Python, & PyTorch/NumPy

ShaunGPT is my implementation of a Generative Pre-trained Transformer (GPT)-style language model. Built using Python and PyTorch/NumPy, this project explores the fundamental architecture and mathematical operations behind transformer-based language models.

The model is trained on the Tiny Shakespeare dataset and learns to generate character-level text by predicting the next character in a sequence.

This educational implementation explores the internal workings of a decoder-only Transformer, including token embeddings, positional encoding, causal self-attention, multi-head attention, feed-forward networks, backpropagation, gradient optimization and autoregressive text generation.

---

## 🎯 Project Objectives

The primary objectives of ShaunGPT are:

- Understand the architecture of GPT-style language models.
- Implement Transformer components using PyTorch.
- Understand causal self-attention and multi-head attention.
- Explore tensor operations and neural network modules.
- Understand forward propagation and automatic differentiation.
- Understand how neural networks learn through backpropagation and gradient descent.
- Train a language model to predict the next character.
- Implement autoregressive text generation.
- Investigate the computational requirements of Transformer architectures.
- Experiment with model scaling, regularization and optimization.

## 🧠 Model Architecture

ShaunGPT uses a decoder-only Transformer architecture inspired by the original GPT design.

| Component | Configuration |
|---|---|
| Architecture | Decoder-only Transformer |
| Programming Language | Python |
| Deep Learning Framework | PyTorch/NumPy |
| Tokenization | Character-level |
| Transformer Blocks | 5 |
| Embedding Dimension | 320 |
| Attention Heads | 5 |
| Head Dimension | 64 |
| Feed-Forward Dimension | 1280 |
| Context Length | 192 |
| Vocabulary | Character-level, dynamically constructed |
| Activation Function | ReLU |
| Normalization | Layer Normalization |
| Positional Encoding | Sinusoidal |
| Attention | Causal Multi-Head Self-Attention |
| Architecture Style | Pre-LayerNorm |
| Optimizer | Adam |
| Loss Function | Cross-Entropy |
| Regularization | Dropout |
| Gradient Clipping | Global Gradient Norm Clipping |
| Text Generation | Autoregressive |
| Numerical Precision | float64 |
| Execution Device | CPU |

## ⚙️ Features

ShaunGPT implements the following components:

- **Character-Level Tokenization:** Constructs a vocabulary from the unique characters present in the training dataset.
- **Token Embeddings:** Converts character IDs into learnable numerical vector representations.
- **Positional Encoding:** Uses sinusoidal positional encodings to provide information about character positions.
- **Layer Normalization:** Normalizes intermediate neural network representations.
- **Causal Self-Attention:** Ensures that the model cannot access future characters when predicting the next character.
- **Multi-Head Attention:** Uses multiple attention heads to learn different relationships between characters.
- **Feed-Forward Network:** Applies nonlinear transformations to the representations produced by attention.
- **Residual Connections:** Helps information and gradients flow through the network.
- **Dropout:** Introduces regularization during training.
- **Cross-Entropy Loss:** Measures the difference between predicted and actual next-character distributions.
- **Adam Optimizer:** Updates model parameters during training.
- **Gradient Clipping:** Limits excessively large gradients to improve training stability.
- **Training and Validation Evaluation:** Measures model performance on separate training and validation datasets.
- **Autoregressive Generation:** Generates text one character at a time using previously generated characters.
- **Temperature Sampling:** Controls the randomness of generated text.
- **Top-k Sampling:** Restricts next-character sampling to the most probable candidates.
- **Technical Reporting:** Reports model configuration, parameter statistics, dataset information and training performance.

## 🏗️ Architecture Overview

```text
                       Input Text
                           |
                           v
                     Character IDs
                           |
                           v
                    Token Embeddings
                           |
                           v
                   Positional Encoding
                           |
                           v
              +----------------------------+
              |    Transformer Block 1     |
              |                            |
              |    Layer Normalization     |
              |    Causal Self-Attention   |
              |    Residual Connection     |
              |    Layer Normalization     |
              |    Feed-Forward Network    |
              |    Residual Connection     |
              +----------------------------+
                           |
                           v
              +----------------------------+
              |    Transformer Block 2     |
              +----------------------------+
                           |
                           v
              +----------------------------+
              |    Transformer Block 3     |
              +----------------------------+
                           |
                           v
              +----------------------------+
              |    Transformer Block 4     |
              +----------------------------+
                           |
                           v
              +----------------------------+
              |    Transformer Block 5     |
              +----------------------------+
                           |
                           v
                   Final LayerNorm
                           |
                           v
                   Output Projection
                           |
                           v
                  Character Probabilities
                           |
                           v
                 Next-Character Prediction
                           |
                           v
                   Autoregressive Loop
                           |
                           v
                    Generated Text
```

## 📚 Dataset

ShaunGPT is trained using the Tiny Shakespeare dataset.

The dataset contains Shakespearean text and provides a compact corpus for experimenting with character-level language modeling.

The dataset is processed using the following pipeline:

1. Load the text dataset.
2. Construct a sorted vocabulary of unique characters.
3. Create character-to-integer and integer-to-character mappings.
4. Encode the complete dataset into integer token IDs.
5. Divide the encoded dataset into training and validation portions.
6. Generate random batches of consecutive character sequences during training.

The default dataset split is:

| Dataset Portion | Percentage |
|---|---:|
| Training | 90% |
| Validation | 10% |

The vocabulary is dynamically constructed from the actual dataset rather than being hardcoded.

## 📊 Training Configuration

The current default training configuration is:

| Parameter | Value |
|---|---:|
| Training Steps | 7,500 |
| Batch Size | 16 |
| Learning Rate | 0.0003 |
| Context Length | 192 |
| Dropout Rate | 0.12 |
| Random Seed | 42 |
| Evaluation Interval | 250 steps |
| Evaluation Batches | 10 |
| Sample Generation Interval | 1,500 steps |
| Generated Characters | 150 |
| Training/Validation Split | 90% / 10% |

The model evaluates training and validation loss periodically throughout training.

It also generates sample text at configured intervals to observe how its character-level predictions develop.

Training performance depends on the dataset size, hardware, numerical precision and model configuration.


## 💻 Hardware and Computational Environment

ShaunGPT was developed and trained on a CPU-based laptop without a dedicated GPU.

**Hardware environment:**

- Processor: Intel Core i5
- GPU: Not used
- Execution Device: CPU
- Programming Language: Python
- Deep Learning Framework: PyTorch/NumPy
- Numerical Precision: float64

The implementation uses CPU-based tensor operations and PyTorch's automatic differentiation system.

This project demonstrates how Transformer architectures can be implemented and experimented with on consumer hardware without requiring a dedicated GPU.

**Understanding Intelligence, One Token at a Time.** 🤖
