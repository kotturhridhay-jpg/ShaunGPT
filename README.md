# ShaunGPT — Understanding Intelligence, One Token at a Time. 🤖

### A GPT-Style Language Model Built from Scratch Using Python and NumPy

ShaunGPT is my first implementation of a Generative Pre-trained Transformer (GPT)-style language model. Built using Python and NumPy, this project explores the fundamental architecture behind modern transformer-based language models.

The model is trained on the Tiny Shakespeare dataset and learns to generate character-level text by predicting the next character in a sequence.

This project is an educational implementation designed to understand how transformer language models work internally — from token embeddings and self-attention to backpropagation and autoregressive text generation.

---

## 🎯 Project Objectives

The primary objectives of ShaunGPT are:

- Understand the architecture of GPT-style language models.
- Implement transformer components using NumPy.
- Understand causal self-attention and multi-head attention.
- Implement forward propagation and backward propagation.
- Understand how neural networks learn through gradient descent.
- Train a language model to predict the next character.
- Implement autoregressive text generation.

## 🧠 Model Architecture

ShaunGPT uses a decoder-only Transformer architecture inspired by the original GPT design.

| Component              | Configuration                    |
|------------------------|----------------------------------|
| Architecture           | Decoder-only Transformer         |  
| Programming Language   | Python                           |
| Numerical Library      | NumPy                            |
| Tokenization           | Character-level                  |
| Transformer Blocks     | 4                                |
| Embedding Dimension    | 256                              |
| Attention Heads        | 4                                |
| Head Dimension         | 64                               |
| Feed-Forward Dimension | 1024                             |                    
| Context Length         | 128                              |
| Vocabulary             | Character-level vocabulary       |
| Activation Function    | ReLU                             |
| Normalization          | Layer Normalization              |
| Positional Encoding    | Sinusoidal                       |
| Attention              | Causal Multi-Head Self-Attention |
| Optimizer              | Adam                             |
| Regularization         | Dropout                          |
| Gradient Clipping      | Global Gradient Clipping         |
| Text Generation        | Autoregressive                   |

## ⚙️ Features

ShaunGPT implements the following components:

- **Token Embeddings:** Converts character IDs into numerical vector representations.
- **Positional Encoding:** Adds information about character positions in a sequence.
- **Layer Normalization:** Normalizes intermediate neural network representations.
- **Causal Self-Attention:** Allows the model to use preceding characters without accessing future characters.
- **Multi-Head Attention:** Uses multiple attention heads to learn different relationships.
- **Feed-Forward Network:** Processes the representations produced by the attention mechanism.
- **Residual Connections:** Helps information and gradients flow through the network.
- **Dropout:** Introduces regularization during training.
- **Cross-Entropy Loss:** Measures the difference between predicted and actual next-character distributions.
- **Adam Optimizer:** Updates model parameters during training.
- **Gradient Clipping:** Limits excessively large gradients.
- **Autoregressive Generation:** Generates text one character at a time.

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
          +---------------------------+
          |   Transformer Block 1     |
          |                           |
          |  Layer Normalization      |
          |  Causal Self-Attention    |
          |  Residual Connection      |
          |  Layer Normalization      |
          |  Feed-Forward Network    |
          |  Residual Connection      |
          +---------------------------+
                      |
                      v
          +---------------------------+
          |   Transformer Block 2     |
          +---------------------------+
                      |
                      v
          +---------------------------+
          |   Transformer Block 3     |
          +---------------------------+
                      |
                      v
          +---------------------------+
          |   Transformer Block 4     |
          +---------------------------+
                      |
                      v
                Output Projection
                      |
                      v
                   Softmax
                      |
                      v
             Next-Character Prediction
                      |
                      v
               Generated Text
```

## 📚 Dataset

ShaunGPT is trained using the Tiny Shakespeare dataset.

The dataset contains Shakespearean text and provides a small corpus for experimenting with character-level language modeling.

The model learns patterns in the training data and uses those patterns to generate new character sequences.

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ShaunGPT.git
```

### 2. Navigate to the Project Directory

```bash
cd ShaunGPT
```

### 3. Install Dependencies

```bash
pip install numpy
```

### 4. Prepare the Dataset

Place the dataset file inside the project directory using the filename expected by the training script.

For example:

```text
ShaunGPT/
│
├── Shaun.py
├── input.txt
└── README.md
```

### 5. Train the Model

```bash
python Shaun.py
```

You can also configure training parameters through the command-line arguments supported by the script.

## 📊 Training Configuration

The current default configuration includes:

| Parameter           | Value    |
|---------------------|----------|
| Training Steps      | 2,000    |
| Batch Size          | 32       |
| Learning Rate       | 0.0005   |
| Context Length      | 128      |
| Dropout Rate        | 0.10     |
| Random Seed         | 42       |
| Evaluation Interval | 50 steps |
| Evaluation Batches  | 10       |

Training performance depends on the hardware, dataset size, and configuration.

## 💻 Hardware

ShaunGPT was developed and trained on a CPU-based laptop.

**Hardware environment:**
- Processor: Intel Core i5, 11th Generation
- GPU: Not used
- Framework: NumPy
- Training: CPU

This project demonstrates that a small educational transformer can be implemented and trained without a dedicated GPU.

## ⚠️ Limitations

ShaunGPT is an educational language model, not a production-ready conversational AI system.
It uses character-level tokenization rather than a modern subword tokenizer.
It has a relatively small architecture.
It is trained on a limited dataset.
Its generated text may contain grammatical inconsistencies and repetition.
It does not have conversational instruction-following capabilities.
Its training speed is limited by CPU hardware.

## 🛣️ Future Improvements
Potential directions for future development include:
Experimenting with larger context lengths.
Improving training efficiency.
Exploring different sampling strategies.
Investigating regularization techniques.
Studying larger transformer architectures.
Exploring subword tokenization.
Building more advanced language models from scratch.
These are possible future directions, not features currently implemented.

## 👨‍💻 About the Developer

ShaunGPT is my first major step into understanding and implementing transformer-based language models.
I am a high-school student interested in Artificial Intelligence, Machine Learning, Deep Learning, and Space Technology.
I developed this project as a hands-on learning experience, using ChatGPT as a programming and learning assistant throughout the process.
My long-term goal is to understand the underlying mathematics and engineering of modern AI systems and progressively build more advanced models from scratch.
