from pathlib import Path  # Imports the Path class from pathlib module for handling file system paths
import argparse          # Imports argparse module for parsing command-line arguments
import math               # Imports math for the square-root scaling used in attention
import time               # Imports Python's time module to measure elapsed runtime
import torch              # Imports PyTorch
import torch.nn as nn     # Imports PyTorch neural-network layers
import torch.nn.functional as F  # Imports PyTorch loss and activation functions

# ========================= Configuration =========================
SEED = 42                # Sets a fixed random seed value for reproducibility
MAX_CHARS = 500_000      # Defines the maximum number of characters to load from the dataset
DEFAULT_STEPS = 5000     # Sets the default number of training iterations
DEFAULT_BATCH_SIZE = 32  # Sets the default number of sequences processed in parallel per batch
DEFAULT_SEQ_LEN = 128    # Sets the default context length (number of tokens per sequence)
DEFAULT_DIM_MODEL = 256  # Sets the default hidden dimension size for the model
DEFAULT_N_HEADS = 4      # Sets the default number of attention heads in multi-head attention
DEFAULT_BLOCKS = 4       # Sets the default number of transformer blocks in the model
DEFAULT_LEARNING_RATE = 0.0005  # Sets the default learning rate for the optimizer
DEFAULT_SAMPLE_EVERY = 1000      # Specifies how often (in steps) to generate and print text samples
DEFAULT_GENERATE_TOKENS = 100   # Sets the default number of new tokens to generate during sampling
DEFAULT_DROPOUT_RATE = 0.10     # Sets the dropout rate

# Keep the original NumPy implementation's float64 parameter precision and run on CPU.
DEVICE = torch.device("cpu")
MODEL_DTYPE = torch.float64

# ========================= Utility functions =========================
def create_causal_mask(seq_len: int, device: torch.device | str | None = None, dtype: torch.dtype = MODEL_DTYPE) -> torch.Tensor:  # Defines a function to create a causal attention mask for autoregressive decoding
    """Return (seq_len, seq_len) additive mask: future positions are -inf."""
    if seq_len <= 0:  # Checks if the sequence length is less than or equal to zero
        raise ValueError("seq_len must be positive")  # Raises an error if sequence length is invalid

    future_positions = torch.triu(
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),
        diagonal=1,
    )  # Marks positions that are strictly above the main diagonal
    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)  # Initializes all allowed positions to zero
    return mask.masked_fill(future_positions, float("-inf"))  # Sets future positions to negative infinity

def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:  # Defines a numerically stable softmax helper
    """Numerically stable softmax using PyTorch."""
    return torch.softmax(logits, dim=dim)  # PyTorch softmax uses a numerically stable implementation

# ========================= Embedding layer =========================
class EmbeddingsLEGO(nn.Module):  # Defines the token embedding layer using PyTorch
    def __init__(self, vocab_size: int, embedding_dim: int):  # Constructor initializing embedding weights
        super().__init__()  # Initializes the parent nn.Module
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # Creates a trainable token-embedding table
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)  # Initializes embeddings with the original small normal distribution

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Converts token IDs to embedding vectors
        return self.embeddings(inputs.long())  # Looks up and returns embedding vectors for the token IDs

# ========================= Linear layer =========================
class LinearLEGO(nn.Module):  # Defines a fully connected linear transformation layer
    def __init__(self, in_features: int, out_features: int, include_bias: bool = True):  # Constructor for weights and optional bias
        super().__init__()  # Initializes the parent nn.Module
        self.linear = nn.Linear(in_features, out_features, bias=include_bias)  # Creates the PyTorch linear layer
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)  # Initializes weights using the original small normal distribution
        if include_bias:  # Checks if bias is included
            nn.init.zeros_(self.linear.bias)  # Initializes the bias vector to zeros

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Forward pass for the linear layer
        return self.linear(inputs)  # Applies the PyTorch linear transformation

# ========================= ReLU activation =========================
class ReLU_LEGO(nn.Module):  # Defines the Rectified Linear Unit (ReLU) activation layer
    def __init__(self):  # Constructor for ReLU layer
        super().__init__()  # Initializes the parent nn.Module
        self.activation = nn.ReLU()  # Creates the PyTorch ReLU activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Forward pass applying ReLU activation
        return self.activation(inputs)  # Returns element-wise max(0, input)

# ========================= Dropout layer =========================
class DropoutLEGO(nn.Module):  # Defines the dropout layer
    def __init__(self, dropout_rate=0.10):  # Constructor for dropout
        super().__init__()  # Initializes the parent nn.Module
        if not 0.0 <= dropout_rate < 1.0:  # Checks that the dropout rate is valid
            raise ValueError("Dropout rate must be between 0 and 1.")  # Raises an exception for an invalid dropout rate
        self.dropout_rate = dropout_rate  # Stores the probability that each activation will be dropped
        self.dropout = nn.Dropout(p=dropout_rate)  # Creates PyTorch dropout; active only in training mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Defines the forward pass
        return self.dropout(inputs)  # Applies inverted dropout during training and identity during evaluation

# ========================= Layer normalization =========================
class LayerNormLEGO(nn.Module):  # Defines the Layer Normalization class
    def __init__(self, input_dim: int, epsilon: float = 1e-5):  # Constructor initializing normalization parameters
        super().__init__()  # Initializes the parent nn.Module
        self.input_dim = input_dim  # Stores the input feature dimension size
        self.epsilon = epsilon  # Stores the small constant used for numerical stability
        self.layer_norm = nn.LayerNorm(input_dim, eps=epsilon)  # Creates learnable scale and shift parameters

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Forward pass for layer normalization
        return self.layer_norm(inputs)  # Normalizes the last dimension and applies learnable scale and bias

# ========================= Sinusoidal position encodings =========================
class PositionalEncodingsLEGO(nn.Module):  # Defines the positional encoding module
    def __init__(self, dim_model: int, max_seq_len: int = 5000):  # Constructor for sinusoidal positional encodings
        super().__init__()  # Initializes the parent nn.Module
        if dim_model <= 0 or max_seq_len <= 0:  # Validates model dimension and maximum sequence length
            raise ValueError("dim_model and max_seq_len must be positive")  # Raises error if parameters are invalid
        self.dim_model = dim_model  # Stores model dimension
        self.max_seq_len = max_seq_len  # Stores maximum sequence length

        pe = torch.zeros((max_seq_len, dim_model), dtype=torch.float64)  # Initializes positional encoding matrix with zeros
        position = torch.arange(max_seq_len, dtype=torch.float64).unsqueeze(1)  # Creates position indices as a column vector
        div_term = torch.exp(
            torch.arange(0, dim_model, 2, dtype=torch.float64)
            * (-math.log(10000.0) / dim_model)
        )  # Computes the frequency terms for sinusoidal encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # Applies sine to even positional-encoding dimensions
        # Handles both even and odd model dimensions safely.
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])  # Applies cosine to odd dimensions
        self.register_buffer("pe", pe.unsqueeze(0))  # Stores encodings as a non-trainable model buffer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Forward pass adding positional encodings to inputs
        seq_len = inputs.shape[1]  # Gets sequence length from input tensor shape
        if seq_len > self.max_seq_len:  # Checks if sequence length exceeds maximum allowed length
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")  # Raises error if sequence is too long
        return inputs + self.pe[:, :seq_len, :]  # Adds positional encodings to token embeddings

# ========================= Softmax layer (used by attention) =========================
class SoftmaxLEGO(nn.Module):  # Defines a Softmax layer class for the attention mechanism
    def __init__(self):  # Constructor for Softmax layer
        super().__init__()  # Initializes the parent nn.Module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Forward pass applying stable softmax
        return stable_softmax(inputs, dim=-1)  # Computes normalized attention probabilities

# ========================= Multi-head causal self-attention =========================
class MultiHeadAttentionLEGO(nn.Module):  # Defines the Multi-Head Causal Self-Attention layer
    def __init__(self, dim_in: int, dim_out: int, n_heads: int):  # Constructor initializing attention projections
        super().__init__()  # Initializes the parent nn.Module
        if n_heads <= 0 or dim_out % n_heads != 0:  # Validates head count and output dimensions
            raise ValueError("dim_out must be divisible by n_heads")  # Raises error if dimensions do not match
        self.dim_in = dim_in  # Stores input dimension
        self.dim_out = dim_out  # Stores output dimension
        self.n_heads = n_heads  # Stores number of attention heads
        self.head_dim = dim_out // n_heads  # Computes dimension size per attention head

        self.query_linear = LinearLEGO(dim_in, dim_out, include_bias=False)  # Linear projection for queries
        self.key_linear = LinearLEGO(dim_in, dim_out, include_bias=False)  # Linear projection for keys
        self.value_linear = LinearLEGO(dim_in, dim_out, include_bias=False)  # Linear projection for values
        self.final_projection = LinearLEGO(dim_out, dim_out, include_bias=False)  # Final attention output projection
        self.softmax = SoftmaxLEGO()  # Softmax layer for attention weights

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:  # Forward pass for multi-head attention
        batch, time_steps, _ = inputs.shape  # Unpacks batch size and sequence length
        queries = self.query_linear(inputs).reshape(batch, time_steps, self.n_heads, self.head_dim)  # Projects and reshapes queries
        keys = self.key_linear(inputs).reshape(batch, time_steps, self.n_heads, self.head_dim)  # Projects and reshapes keys
        values = self.value_linear(inputs).reshape(batch, time_steps, self.n_heads, self.head_dim)  # Projects and reshapes values

        queries = queries.transpose(1, 2)  # Changes query shape to (batch, heads, time, head_dim)
        keys = keys.transpose(1, 2)  # Changes key shape to (batch, heads, time, head_dim)
        values = values.transpose(1, 2)  # Changes value shape to (batch, heads, time, head_dim)

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)  # Computes scaled dot-product attention scores
        if mask is not None:  # Checks if a causal attention mask is provided
            scores = scores + mask.unsqueeze(0).unsqueeze(0)  # Adds the causal mask to block attention to future tokens
        attention_weights = self.softmax(scores)  # Applies softmax to obtain attention probabilities

        attended = attention_weights @ values  # Computes the weighted sum of value vectors
        attended = attended.transpose(1, 2).contiguous().reshape(batch, time_steps, self.dim_out)  # Combines all attention heads
        return self.final_projection(attended)  # Applies the final attention projection

# ========================= Feed-forward network =========================
class FeedForwardNetworkLEGO(nn.Module):  # Defines the position-wise Feed-Forward Network
    def __init__(self, input_units: int, hidden_units: int, output_units: int, dropout_rate=0.10):  # Constructor initializing FFN layers
        super().__init__()  # Initializes the parent nn.Module
        self.linear1 = LinearLEGO(input_units, hidden_units)  # First linear expansion layer
        self.activation = ReLU_LEGO()  # ReLU activation layer
        self.dropout = DropoutLEGO(dropout_rate)  # Dropout layer
        self.linear2 = LinearLEGO(hidden_units, output_units)  # Second linear projection back to model dimension

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Forward pass through the feed-forward network
        x = self.linear1(inputs)  # Projects input features into the larger hidden dimension
        x = self.activation(x)  # Applies ReLU to hidden activations
        x = self.dropout(x)  # Applies dropout when the model is in training mode
        return self.linear2(x)  # Projects hidden activations back to the model dimension

# ========================= Transformer decoder block =========================
class TransformerBlockLEGO(nn.Module):  # Defines a single Transformer Decoder Block
    """Pre-layer-normalized transformer block with residual connections."""
    def __init__(self, dim_model: int, n_heads: int):  # Constructor initializing attention, layer norms, and FFN
        super().__init__()  # Initializes the parent nn.Module
        self.multi_head_attention = MultiHeadAttentionLEGO(dim_model, dim_model, n_heads)  # Multi-head attention sub-layer
        self.layernorm1 = LayerNormLEGO(dim_model)  # First layer normalization applied before attention (Pre-LN)
        self.ffn = FeedForwardNetworkLEGO(  # Creates the position-wise feed-forward network
            dim_model,  # Input feature dimension
            4 * dim_model,  # Hidden dimension is four times the model dimension
            dim_model,  # Output dimension returns to the model dimension
            dropout_rate=DEFAULT_DROPOUT_RATE,  # Uses the configured dropout rate
        )  # Completes feed-forward network construction
        self.layernorm2 = LayerNormLEGO(dim_model)  # Second layer normalization applied before FFN (Pre-LN)

    def forward(
        self,
        inputs: torch.Tensor,
        mask_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:  # Forward pass for transformer block
        normalized1 = self.layernorm1(inputs)  # Applies first layer normalization
        attention_output = self.multi_head_attention(normalized1, mask=mask_attention)  # Applies causal self-attention
        mid = inputs + attention_output  # Adds residual connection around the attention sub-layer
        normalized2 = self.layernorm2(mid)  # Applies second layer normalization
        ffn_output = self.ffn(normalized2)  # Applies the feed-forward network
        return mid + ffn_output  # Adds residual connection around the feed-forward sub-layer

# ========================= Cross-entropy loss =========================
class CrossEntropyWithSoftmaxLEGO(nn.Module):  # Defines the combined softmax and cross-entropy loss
    def __init__(self):  # Constructor for the loss module
        super().__init__()  # Initializes the parent nn.Module

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # Computes cross-entropy loss
        if logits.ndim != 3:  # Validates that logits have shape (batch, time, vocabulary)
            raise ValueError("logits must have shape (batch, time, vocabulary)")  # Raises an error for an incorrect shape
        batch, time_steps, vocab_size = logits.shape  # Unpacks logits dimensions
        labels_flat = labels.long().reshape(-1)  # Flattens target labels and converts them to integer indices
        if labels_flat.numel() != batch * time_steps:  # Validates target count against logits dimensions
            raise ValueError("labels do not match logits batch/time dimensions")  # Raises an error if sizes do not match
        if torch.any(labels_flat < 0) or torch.any(labels_flat >= vocab_size):  # Checks for out-of-range vocabulary indices
            raise ValueError("label index is outside the vocabulary")  # Raises an error if any label is invalid

        flat_logits = logits.reshape(-1, vocab_size)  # Reshapes logits to (batch * time, vocabulary)
        return F.cross_entropy(flat_logits, labels_flat, reduction="mean")  # Computes mean cross-entropy; autograd calculates gradients

# ========================= GPT model =========================
class GPTLEGO(nn.Module):  # Defines the complete character-level GPT model
    def __init__(
        self,
        vocab_size: int,
        blocks: int,
        dim_model: int,
        n_heads: int,
        max_seq_len: int = 5000,
    ):  # Constructor initializing the GPT architecture
        super().__init__()  # Initializes the parent nn.Module
        if vocab_size <= 1:  # Validates vocabulary size
            raise ValueError("vocab_size must be greater than 1")  # Raises error if vocabulary is too small
        if blocks <= 0:  # Validates block count
            raise ValueError("blocks must be positive")  # Raises error if block count is invalid
        if n_heads <= 0 or dim_model <= 0 or dim_model % n_heads != 0:  # Validates model dimension and head divisibility
            raise ValueError("dim_model must be positive and divisible by n_heads")  # Raises error if dimensions are invalid

        self.vocab_size = vocab_size  # Stores vocabulary size
        self.embedding_decoder = EmbeddingsLEGO(vocab_size, dim_model)  # Initializes token embedding layer
        self.positional_encodings_decoder = PositionalEncodingsLEGO(dim_model, max_seq_len)  # Initializes positional encodings
        self.decoder_blocks = nn.ModuleList(  # Registers all transformer blocks with PyTorch
            [TransformerBlockLEGO(dim_model, n_heads) for _ in range(blocks)]
        )  # Completes the decoder-block list
        self.final_layernorm = LayerNormLEGO(dim_model)  # Final layer normalization before output projection
        self.final_linear = LinearLEGO(dim_model, vocab_size)  # Projects hidden states to vocabulary logits
        self.loss_function = CrossEntropyWithSoftmaxLEGO()  # Creates the cross-entropy loss module

    def forward(
        self,
        input_tokens_decoder: torch.Tensor,
        mask_attention: torch.Tensor | None = None,
    ) -> torch.Tensor:  # Forward pass for the full GPT model
        x = self.embedding_decoder(input_tokens_decoder)  # Converts token IDs to embedding vectors
        x = self.positional_encodings_decoder(x)  # Adds positional encodings to token embeddings
        for decoder in self.decoder_blocks:  # Iterates through each transformer block
            x = decoder(x, mask_attention=mask_attention)  # Passes hidden states through the transformer block
        x = self.final_layernorm(x)  # Applies final layer normalization
        return self.final_linear(x)  # Projects hidden states to vocabulary logits

    @torch.no_grad()  # Disables gradient tracking during autoregressive generation
    def generate(
        self,
        start_tokens,
        max_new_tokens: int,
        seq_len: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[int]:  # Autoregressive text generation method
        """Generate max_new_tokens autoregressively; start_tokens are character IDs."""
        if temperature <= 0:  # Validates temperature
            raise ValueError("temperature must be greater than zero")  # Raises error if temperature is non-positive
        if seq_len <= 0 or max_new_tokens < 0:  # Validates sequence length and new token count
            raise ValueError("seq_len must be positive and max_new_tokens cannot be negative")  # Raises error if invalid

        generated = list(map(int, start_tokens))  # Initializes generated token list with starting token IDs
        if not generated:  # Checks if starting tokens are empty
            raise ValueError("start_tokens must contain at least one token")  # Raises error if no start tokens were provided

        was_training = self.training  # Stores the model's current training/evaluation mode
        self.eval()  # Disables dropout during text generation
        device = next(self.parameters()).device  # Gets the device containing model parameters
        dtype = next(self.parameters()).dtype  # Gets the model parameter data type

        try:  # Ensures the original model mode is restored after generation
            for _ in range(max_new_tokens):  # Generates new tokens one at a time
                context = generated[-seq_len:]  # Extracts the latest context up to the configured sequence length
                x_input = torch.tensor([context], dtype=torch.long, device=device)  # Converts context to a PyTorch token tensor
                mask = create_causal_mask(len(context), device=device, dtype=dtype)  # Creates the causal attention mask
                logits = self(x_input, mask_attention=mask)  # Computes the logits for all context positions
                next_logits = logits[0, -1, :] / temperature  # Extracts the last position's logits and applies temperature

                if top_k is not None:  # Checks if top-k sampling is enabled
                    if top_k <= 0:  # Validates top_k value
                        raise ValueError("top_k must be positive when provided")  # Raises an error if top_k is invalid
                    k = min(top_k, next_logits.numel())  # Ensures k does not exceed vocabulary size
                    keep_indices = torch.topk(next_logits, k).indices  # Finds indices of the k highest-scoring logits
                    filtered = torch.full_like(next_logits, float("-inf"))  # Initializes all filtered logits to negative infinity
                    filtered[keep_indices] = next_logits[keep_indices]  # Retains only the top-k logits
                    next_logits = filtered  # Uses the filtered logits for sampling

                probabilities = stable_softmax(next_logits, dim=-1)  # Computes the probability distribution over vocabulary
                next_token = int(torch.multinomial(probabilities, num_samples=1).item())  # Samples the next token ID
                generated.append(next_token)  # Appends the new token to the sequence
        finally:  # Restores the model mode even if generation raises an error
            self.train(was_training)  # Returns the model to its previous training/evaluation mode

        return generated  # Returns the full list of generated token IDs

# ========================= Dataset loading =========================
def load_shakespeare(data_path: str | Path | None = None, max_chars: int | None = MAX_CHARS):  # Defines dataset loading utility function
    """Read the local input.txt; no network download is required."""
    if data_path is None:  # Checks if data path is not specified
        data_path = Path(__file__).resolve().with_name("input.txt")  # Defaults to input.txt located next to this script
    data_path = Path(data_path).expanduser().resolve()  # Resolves the absolute path to the dataset file
    if not data_path.is_file():  # Checks if dataset file exists
        raise FileNotFoundError(  # Raises a file-not-found error with a helpful troubleshooting message
            f"Could not find dataset: {data_path}\n"
            "Place input.txt in the same folder as Shaun_fixed.py, or pass --data PATH."
        )  # Closes the error constructor call

    text = data_path.read_text(encoding="utf-8")  # Reads the text file using UTF-8 encoding
    if not text:  # Checks if text file is empty
        raise ValueError(f"Dataset is empty: {data_path}")  # Raises an error if dataset has no characters
    if max_chars is not None:  # Checks if maximum character limit is set
        if max_chars <= 0:  # Validates max_chars value
            raise ValueError("max_chars must be positive or None")  # Raises an error if invalid
        text = text[:max_chars]  # Truncates text to the maximum specified character count

    chars = sorted(set(text))  # Extracts sorted unique characters in the dataset
    char_to_int = {ch: i for i, ch in enumerate(chars)}  # Creates character-to-integer mapping
    int_to_char = {i: ch for ch, i in char_to_int.items()}  # Creates integer-to-character mapping
    encoded_data = torch.tensor([char_to_int[ch] for ch in text], dtype=torch.long)  # Encodes text as PyTorch integer token IDs
    if len(encoded_data) < 2:  # Validates encoded dataset length
        raise ValueError("Dataset needs at least two characters to create input/target pairs")  # Raises an error if dataset is too short
    return encoded_data, len(chars), char_to_int, int_to_char, text  # Returns encoded data, vocabulary size, mappings, and raw text

def get_batch(data: torch.Tensor, batch_size: int, seq_len: int):  # Defines function to sample random training batches
    """Sample random contiguous input/next-character target pairs."""
    if len(data) <= seq_len:  # Checks if dataset is shorter than sequence length
        raise ValueError(f"Dataset must contain more than seq_len={seq_len} characters")  # Raises an error if dataset is too short
    offsets = torch.randint(0, len(data) - seq_len, (batch_size,))  # Samples random starting offsets for batch sequences
    x = torch.stack([data[i:i + seq_len] for i in offsets.tolist()])  # Creates input sequences of length seq_len
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in offsets.tolist()])  # Creates next-character targets shifted by one
    return x, y  # Returns input batch x and target batch y

# ========================= Validation and checkpointing =========================
def estimate_loss(model, train_data, val_data, batch_size, seq_len, mask, eval_batches=10):  # Estimates train and validation losses
    """Estimate train/validation loss on fresh batches without updating weights."""
    results = {}  # Creates a dictionary to hold the estimated loss for each dataset split
    was_training = model.training  # Stores the model's current mode
    model.eval()  # Disables dropout for evaluation

    try:  # Ensures the original model mode is restored
        with torch.no_grad():  # Disables gradient calculation during evaluation
            for split_name, split_data in (("train", train_data), ("val", val_data)):  # Loops over training and validation datasets
                losses = []  # Creates a list to collect evaluation loss values
                for _ in range(eval_batches):  # Repeats evaluation over the requested number of batches
                    x_batch, y_batch = get_batch(split_data, batch_size, seq_len)  # Samples input and target sequences
                    logits = model(x_batch, mask_attention=mask)  # Runs the model with the causal attention mask
                    losses.append(model.loss_function(logits, y_batch))  # Computes and stores batch cross-entropy loss
                results[split_name] = torch.stack(losses).mean().item()  # Stores average loss for the current dataset split
    finally:  # Restores the model's previous mode
        model.train(was_training)  # Returns the model to its original training/evaluation mode

    return results  # Returns training and validation loss estimates

def main():  # Defines the main function that configures and runs model training
    parser = argparse.ArgumentParser(  # Creates a command-line argument parser for training options
        description="Train a small PyTorch character-level GPT on Tiny Shakespeare, with validation."
    )  # Closes the argument parser constructor call
    parser.add_argument("--data", type=Path, default=None,  # Adds an optional command-line argument for the dataset file path
                        help="Path to input.txt (default: beside this script)")  # Explains the dataset path argument
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,  # Adds the number of optimizer steps as a command-line option
                        help="Number of training steps")  # Explains the training-step argument
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,  # Adds the number of sequences per training batch as an option
                        help="Sequences per training batch")  # Explains the batch-size argument
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,  # Adds the context length as a command-line option
                        help="Context length")  # Explains the sequence-length argument
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY,  # Adds an option controlling how often generated samples are printed
                        help="Generate text every N steps; 0 disables samples")  # Explains the sample interval
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,  # Adds an option limiting loaded dataset characters
                        help="Maximum dataset characters; use 0 for the entire file")  # Explains the dataset-size limit
    parser.add_argument("--val-fraction", type=float, default=0.10,  # Adds the fraction reserved for validation
                        help="Fraction of the final dataset reserved for validation (default: 0.10)")  # Explains the validation split
    parser.add_argument("--eval-every", type=int, default=250,  # Adds the interval between loss evaluations
                        help="Evaluate train/validation loss every N steps")  # Explains the evaluation interval
    parser.add_argument("--eval-batches", type=int, default=10,  # Adds the number of batches averaged during each evaluation
                        help="Number of batches averaged for each loss evaluation")  # Explains the evaluation batch-count argument
    args = parser.parse_args()  # Reads the command-line arguments supplied when the script is launched

    if args.steps <= 0 or args.batch_size <= 0 or args.seq_len <= 0:  # Checks that dimensions and step count are positive
        parser.error("--steps, --batch-size and --seq-len must all be positive")  # Reports invalid training dimensions
    if args.sample_every < 0:  # Checks that the sampling interval is not negative
        parser.error("--sample-every cannot be negative")  # Reports an invalid sampling interval
    if args.eval_every <= 0 or args.eval_batches <= 0:  # Checks that evaluation settings are positive
        parser.error("--eval-every and --eval-batches must be positive")  # Reports invalid evaluation settings
    if not 0.0 < args.val_fraction < 0.5:  # Ensures the validation fraction is between zero and one half
        parser.error("--val-fraction must be between 0 and 0.5")  # Reports an invalid validation fraction

    torch.manual_seed(SEED)  # Seeds PyTorch's random-number generator for repeatable initialization and sampling
    max_chars = None if args.max_chars == 0 else args.max_chars  # Uses the full file when the limit is zero
    data, vocab_size, char_to_int, int_to_char, source_text = load_shakespeare(args.data, max_chars)  # Loads and encodes the text dataset
    split_index = int(len(data) * (1.0 - args.val_fraction))  # Calculates the index where the training portion ends
    train_data = data[:split_index]  # Assigns the earlier portion of encoded text to training
    val_data = data[split_index:]  # Assigns the final portion of encoded text to validation
    if len(train_data) <= args.seq_len or len(val_data) <= args.seq_len:  # Checks that both splits can provide full sequences and targets
        parser.error("Training and validation splits must each be longer than --seq-len")  # Reports an invalid dataset split

    print(f"Dataset: {len(data):,} characters | vocabulary: {vocab_size} unique characters")  # Prints dataset size and vocabulary size
    print(f"Source: {Path(args.data).resolve() if args.data else Path(__file__).resolve().with_name('input.txt')}")  # Prints dataset path
    print(f"Split: {len(train_data):,} train characters ({1-args.val_fraction:.0%}) | "  # Begins printing the sizes and percentages of the training and validation splits
          f"{len(val_data):,} validation characters ({args.val_fraction:.0%})")  # Completes the dataset split summary

    dim_model = DEFAULT_DIM_MODEL  # Selects the configured transformer hidden dimension
    n_heads = DEFAULT_N_HEADS  # Selects the configured number of attention heads
    blocks = DEFAULT_BLOCKS  # Selects the configured number of transformer decoder blocks
    model = GPTLEGO(  # Constructs the character-level GPT using the selected architecture settings
        vocab_size,
        blocks,
        dim_model,
        n_heads,
        max_seq_len=max(5000, args.seq_len),
    ).to(device=DEVICE, dtype=MODEL_DTYPE)  # Places the model on CPU and preserves float64 precision
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)  # Creates the PyTorch Adam optimizer
    mask = create_causal_mask(args.seq_len, device=DEVICE, dtype=MODEL_DTYPE)  # Builds the causal mask for training sequences

    print(f"Building ShaunGPT: {blocks} block(s), d_model={dim_model}, heads={n_heads}, context={args.seq_len}")  # Displays model architecture
    print(f"Training for {args.steps} steps...\n")  # Displays the planned number of training steps

    for step in range(1, args.steps + 1):  # Runs the training loop from step one through the requested final step
        model.train()  # Enables dropout and training behavior
        x_batch, y_batch = get_batch(train_data, args.batch_size, args.seq_len)  # Samples training inputs and next-character targets
        optimizer.zero_grad(set_to_none=True)  # Clears gradients from the previous training step
        logits = model(x_batch, mask_attention=mask)  # Computes vocabulary scores for each input position
        train_loss = model.loss_function(logits, y_batch)  # Calculates the average next-character prediction loss
        train_loss.backward()  # Uses PyTorch autograd to backpropagate through the complete GPT model
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Applies the original global gradient-norm clipping limit
        optimizer.step()  # Updates all model parameters using Adam

        if step == 1 or step % 50 == 0 or step == args.steps:  # Prints training loss on the first step, every 50 steps, and the final step
            print(f"Step [{step:04d}/{args.steps:04d}] | training loss: {train_loss.item():.4f}")  # Displays step number and training loss

        should_evaluate = (step == 1 or step % args.eval_every == 0 or step == args.steps)  # Checks whether evaluation is due
        if should_evaluate:  # Runs evaluation only when scheduled
            losses = estimate_loss(  # Estimates loss on fresh training and validation batches
                model,
                train_data,
                val_data,
                args.batch_size,
                args.seq_len,
                mask,
                args.eval_batches,
            )  # Completes the evaluation call
            current_train = losses["train"]  # Extracts estimated training loss
            current_val = losses["val"]  # Extracts estimated validation loss
            print(f"  Evaluation | train loss: {current_train:.4f} | validation loss: {current_val:.4f}")  # Prints average training and validation losses

        if args.sample_every and (step % args.sample_every == 0 or step == args.steps):  # Checks whether sampling is enabled and due
            prompt = "First Citizen:"  # Sets the text prompt used to start sample generation
            prompt_tokens = [char_to_int[ch] for ch in prompt if ch in char_to_int]  # Converts available prompt characters to token IDs
            if not prompt_tokens:  # Checks whether the prompt produced any valid vocabulary tokens
                prompt_tokens = [int(data[0].item())]  # Uses the dataset's first token as a fallback
            generated_tokens = model.generate(  # Starts autoregressive generation from the prompt token IDs
                prompt_tokens,
                max_new_tokens=DEFAULT_GENERATE_TOKENS,
                seq_len=args.seq_len,
                temperature=0.8,
                top_k=min(20, vocab_size),
            )  # Completes the text-generation call
            generated_text = "".join(int_to_char[token] for token in generated_tokens)  # Converts generated IDs back into characters
            print("\n--- ShaunGPT sample ---")  # Prints a heading before generated text
            print(generated_text)  # Displays the generated character sequence
            print("-----------------------\n")  # Prints a separator after the generated sample

    print("ShaunGPT Training successfully finished!")  # Displays a message indicating training has ended

if __name__ == "__main__":  # Runs training only when this file is executed directly
    start_time = time.time()  # Records the start time before training begins
    main()  # Calls the main training function
    elapsed_time = time.time() - start_time  # Calculates total elapsed runtime
    print(f"\nShaunGPT took {elapsed_time:.2f} seconds to train and test itself.")  # Prints total runtime in seconds
