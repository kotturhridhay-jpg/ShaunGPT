from pathlib import Path  # Imports the Path class from pathlib module for handling file system paths
import argparse          # Imports argparse module for parsing command-line arguments
import math               # Imports math for the square-root scaling used in attention
import time               # Imports Python's time module to measure elapsed runtime
import torch              # Imports PyTorch
import torch.nn as nn     # Imports PyTorch neural-network layers
import torch.nn.functional as F  # Imports PyTorch loss and activation functions

# ========================= Configuration =========================
SEED = 42                       # Sets the random seed to 42 so that random operations, such as weight initialization and data sampling, produce reproducible results across runs.
MAX_CHARS = 750_000             # Sets the maximum number of characters to load from the training dataset, allowing the model to train on up to 750,000 characters.
DEFAULT_STEPS = 7500            # Sets the default number of training iterations to 7,500, giving the model more opportunities to update its weights.
DEFAULT_BATCH_SIZE = 32         # Sets the number of training sequences processed together in one training step; a smaller batch helps manage memory usage.
DEFAULT_SEQ_LEN = 192           # Sets the context length to 192 tokens, meaning the model can consider up to 192 previous characters when predicting the next character.
DEFAULT_DIM_MODEL = 320         # Sets the model's embedding and hidden representation size to 320 dimensions, increasing the amount of information each token can represent.
DEFAULT_N_HEADS = 5             # Sets the number of parallel self-attention heads to 5, allowing the model to learn different relationships between tokens simultaneously.
DEFAULT_BLOCKS = 5              # Sets the number of Transformer blocks to 5, increasing the depth of the network and the number of processing stages applied to the input.
DEFAULT_LEARNING_RATE = 0.0003  # Sets the learning rate to 0.0003, controlling how large each optimizer update is when adjusting the model's trainable parameters.
DEFAULT_SAMPLE_EVERY = 1500     # Sets the interval between generated training samples to 1,500 steps, reducing how frequently the model pauses to display generated text.
DEFAULT_GENERATE_TOKENS = 150   # Sets the number of new characters generated in each sample to 150, allowing longer text samples to be inspected.
DEFAULT_DROPOUT_RATE = 0.12     # Sets the dropout probability to 12%, randomly disabling a fraction of activations during training to help reduce overfitting.

# Keep the original NumPy implementation's float64 parameter precision and run on CPU.
DEVICE = torch.device("cpu")  # Creates a PyTorch device object that forces all model operations to run on the CPU.
MODEL_DTYPE = torch.float64   # Sets model parameters and floating-point calculations to 64-bit precision.

# ========================= Utility functions =========================
def create_causal_mask(seq_len: int, device: torch.device | str | None = None, dtype: torch.dtype = MODEL_DTYPE) -> torch.Tensor:  # Defines a function to create a causal attention mask for autoregressive decoding
    """Return (seq_len, seq_len) additive mask: future positions are -inf."""
    if seq_len <= 0:  # Checks if the sequence length is less than or equal to zero
        raise ValueError("seq_len must be positive")  # Raises an error if sequence length is invalid

    future_positions = torch.triu(  # Begins constructing a Boolean matrix that identifies positions later than each token.
        torch.ones((seq_len, seq_len), dtype=torch.bool, device=device),  # Creates a square Boolean matrix of the requested sequence length on the selected device.
        diagonal=1,  # Sets the diagonal offset to 1, so the main diagonal is excluded and only future positions are marked.
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
        div_term = torch.exp(  # Begins calculating the frequency scale used by the sinusoidal position encodings.
            torch.arange(0, dim_model, 2, dtype=torch.float64)  # Creates the even-numbered dimension indices: 0, 2, 4, and so on.
            * (-math.log(10000.0) / dim_model)  # Scales those indices by the negative logarithmic frequency factor.
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

    def forward(  # Begins the attention layer's forward method, which processes the input tensors.
        self,  # Receives the current module instance automatically.
        inputs: torch.Tensor,  # Receives the input tensor containing the token representations.
        mask: torch.Tensor | None = None,  # Optionally receives the mask that prevents attention to future tokens.
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

    def forward(  # Begins the Transformer block's forward method.
        self,  # Receives the current module instance automatically.
        inputs: torch.Tensor,  # Receives the hidden-state tensor entering this Transformer block.
        mask_attention: torch.Tensor | None = None,  # Optionally receives the causal mask to pass into self-attention.
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
    def __init__(  # Begins the GPT model constructor and its list of architecture settings.
        self,  # Receives the current model instance automatically.
        vocab_size: int,  # Specifies how many distinct token IDs the model can represent.
        blocks: int,  # Specifies how many Transformer decoder blocks to stack.
        dim_model: int,  # Specifies the width of each token's hidden representation.
        n_heads: int,  # Specifies how many parallel attention heads each block uses.
        max_seq_len: int = 5000,  # Sets the maximum sequence length supported by positional encodings.
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
            [TransformerBlockLEGO(dim_model, n_heads) for _ in range(blocks)]  # Creates the requested number of independent Transformer blocks using a list comprehension.
        )  # Completes the decoder-block list
        self.final_layernorm = LayerNormLEGO(dim_model)  # Final layer normalization before output projection
        self.final_linear = LinearLEGO(dim_model, vocab_size)  # Projects hidden states to vocabulary logits
        self.loss_function = CrossEntropyWithSoftmaxLEGO()  # Creates the cross-entropy loss module

    def forward(  # Begins the full GPT model's forward pass.
        self,  # Receives the current model instance automatically.
        input_tokens_decoder: torch.Tensor,  # Receives the integer token IDs that form the input sequence.
        mask_attention: torch.Tensor | None = None,  # Optionally receives the causal attention mask.
    ) -> torch.Tensor:  # Forward pass for the full GPT model
        x = self.embedding_decoder(input_tokens_decoder)  # Converts token IDs to embedding vectors
        x = self.positional_encodings_decoder(x)  # Adds positional encodings to token embeddings
        for decoder in self.decoder_blocks:  # Iterates through each transformer block
            x = decoder(x, mask_attention=mask_attention)  # Passes hidden states through the transformer block
        x = self.final_layernorm(x)  # Applies final layer normalization
        return self.final_linear(x)  # Projects hidden states to vocabulary logits

    @torch.no_grad()  # Disables gradient tracking during autoregressive generation
    def generate(  # Begins the autoregressive text-generation method.
        self,  # Receives the current model instance automatically.
        start_tokens,  # Contains the starting token IDs that seed generation.
        max_new_tokens: int,  # Sets how many additional tokens to generate.
        seq_len: int,  # Sets the maximum context length used for each prediction.
        temperature: float = 1.0,  # Controls randomness in sampling; lower values make distributions more concentrated.
        top_k: int | None = None,  # Optionally restricts sampling to the highest-scoring k tokens.
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
            f"Could not find dataset: {data_path}\n"  # Begins the first line of the error message and inserts the missing file's path.
            "Place input.txt in the same folder as Shaun_fixed.py, or pass --data PATH."  # Adds a second troubleshooting line explaining where to put the dataset or how to specify it.
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

# ========================= GPT Technical Report Engine =========================
def print_technical_report(model, args, data, train_data, val_data, vocab_size, initial_losses, final_losses, elapsed_time, data_path):  # Defines the function that prints model, dataset, training, performance, and runtime details.
    # Calculate parameter statistics
    total_parameters = sum(  # Begins summing the number of elements across all model parameter tensors.
        parameter.numel()  # Gets the number of scalar elements in the current parameter tensor.
        for parameter in model.parameters()  # Iterates over every parameter tensor registered in the model.
    )  # Ends the sum expression and stores the total parameter count.

    trainable_parameters = sum(  # Begins counting only parameters that are trainable.
        parameter.numel()  # Gets the number of scalar elements in the current parameter tensor.
        for parameter in model.parameters()  # Iterates over every parameter tensor in the model.
        if parameter.requires_grad  # Includes a parameter only when PyTorch is allowed to calculate gradients for it.
    )  # Ends the sum expression for trainable parameters.

    non_trainable_parameters = total_parameters - trainable_parameters  # Calculates the parameter count that is not trainable by subtracting trainable from total.

    # Calculate model parameter memory
    parameter_memory = sum(  # Begins calculating the storage occupied by model parameter tensors.
        parameter.numel() * parameter.element_size()  # Calculates bytes used by one parameter tensor: element count multiplied by bytes per element.
        for parameter in model.parameters()  # Iterates through all model parameter tensors.
    )  # Ends the parameter-memory calculation.

    buffer_memory = sum(  # Begins calculating memory occupied by registered model buffers.
        buffer.numel() * buffer.element_size()  # Calculates the bytes occupied by the current buffer tensor.
        for buffer in model.buffers()  # Iterates through buffers, such as the fixed positional-encoding matrix.
    )  # Ends the buffer-memory calculation.

    # Extract architecture information
    dim_model = DEFAULT_DIM_MODEL  # Reads the configured model hidden dimension for display in the report.
    n_heads = DEFAULT_N_HEADS  # Reads the configured number of attention heads for display.
    blocks = DEFAULT_BLOCKS  # Reads the configured number of Transformer blocks for display.
    head_dim = dim_model // n_heads  # Calculates the width of each attention head using integer division.
    feed_forward_dim = dim_model * 4  # Calculates the feed-forward network's hidden width, set to four times the model width.

    # Extract runtime information
    device = next(model.parameters()).device  # Gets the device (CPU or GPU) used by the model's first parameter tensor.
    dtype = next(model.parameters()).dtype  # Gets the numerical data type used by the model's first parameter tensor.

    # Print report
    print("\n")  # Prints a blank line before the report.
    print("=" * 65)  # Prints a 65-character divider to make the report visually distinct.
    print("                     SHAUNGPT")  # Prints the model name as the report heading.
    print("                TECHNICAL MODEL REPORT")  # Prints the report subtitle.
    print("=" * 65)  # Prints the closing divider for the heading.

    # 01 - Model identification
    print("\n---------- 01. MODEL IDENTIFICATION ----------")  # Prints the heading for the model-identification section.
    print(f"Model Name          : ShaunGPT")  # Prints the model's name.
    print(f"Architecture        : Decoder-Only Transformer")  # Prints the high-level Transformer architecture type.
    print(f"Model Type          : Character-Level Language Model")  # Identifies the model as predicting individual characters rather than whole words or subwords.
    print(f"Framework           : PyTorch {torch.__version__}")  # Prints the installed PyTorch version using its runtime version string.
    print(f"Execution Device    : {str(device).upper()}")  # Prints the execution device in uppercase, such as CPU.
    print(f"Numerical Precision : {dtype}")  # Prints the model parameter precision, such as torch.float64.

    # 02 - Architecture
    print("\n---------- 02. MODEL ARCHITECTURE ----------")  # Prints the heading for the architecture section.
    print(f"Transformer Blocks  : {blocks}")  # Prints the number of stacked Transformer blocks.
    print(f"Embedding Dimension : {dim_model}")  # Prints the size of each hidden embedding vector.
    print(f"Attention Heads     : {n_heads}")  # Prints the number of attention heads in each block.
    print(f"Head Dimension      : {head_dim}")  # Prints the number of dimensions assigned to each attention head.
    print(f"Context Length      : {args.seq_len}")  # Prints the maximum input context length used for training.
    print(f"Vocabulary Size     : {vocab_size}")  # Prints the number of unique characters represented in the vocabulary.
    print(f"Feed Forward Dim    : {feed_forward_dim}")  # Prints the hidden width of the feed-forward network.
    print(f"Activation          : ReLU")  # Prints the activation function used inside the feed-forward network.
    print(f"Normalization       : Layer Normalization")  # Prints the normalization method used in Transformer blocks.
    print(f"Position Encoding   : Sinusoidal")  # Prints the method used to encode token positions.
    print(f"Attention           : Causal Self-Attention")  # Prints the attention type, which blocks access to future tokens.
    print(f"Residual Connections: Enabled")  # Reports that residual connections are part of the architecture.

    # 03 - Parameter statistics
    print("\n---------- 03. MODEL PARAMETERS ----------")  # Prints the heading for the parameter statistics section.
    print(f"Total Parameters    : {total_parameters:,}")  # Prints the total number of scalar parameters, with comma separators.
    print(f"Trainable Parameters: {trainable_parameters:,}")  # Prints the number of parameters that receive gradient updates.
    print(f"Non-Trainable Params: {non_trainable_parameters:,}")  # Prints the number of parameters that do not receive gradient updates.
    print(f"Parameter Memory    : {parameter_memory / (1024**2):.2f} MiB")  # Converts parameter storage from bytes to mebibytes and prints two decimal places.
    print(f"Buffer Memory       : {buffer_memory / (1024**2):.2f} MiB")  # Converts registered-buffer storage from bytes to mebibytes and prints two decimal places.

    # 04 - Dataset information
    print("\n---------- 04. DATASET INFORMATION ----------")  # Prints the heading for the dataset section.
    print(f"Dataset             : Tiny Shakespeare")  # Prints the dataset's human-readable name.
    print(f"Dataset Source      : {data_path}")  # Prints the resolved path of the source text file.
    print(f"Characters Loaded   : {len(data):,}")  # Prints the number of characters loaded into the encoded dataset.
    print(f"Unique Characters   : {vocab_size}")  # Prints the number of unique characters in the vocabulary.
    print(f"Training Characters : {len(train_data):,}")  # Prints the number of encoded characters assigned to training.
    print(f"Validation Characters: {len(val_data):,}")  # Prints the number of encoded characters assigned to validation.
    print(f"Training Split      : {1 - args.val_fraction:.0%}")  # Prints the percentage of the dataset used for training.
    print(f"Validation Split    : {args.val_fraction:.0%}")  # Prints the percentage of the dataset reserved for validation.

    # 05 - Training configuration
    print("\n---------- 05. TRAINING CONFIGURATION ----------")  # Prints the heading for the training-configuration section.
    print(f"Training Steps      : {args.steps:,}")  # Prints the requested number of optimizer update steps.
    print(f"Batch Size          : {args.batch_size}")  # Prints how many sequences are processed together in each batch.
    print(f"Sequence Length     : {args.seq_len}")  # Prints the number of characters/tokens in each training sequence.
    print(f"Learning Rate       : {DEFAULT_LEARNING_RATE}")  # Prints the learning rate supplied to Adam.
    print(f"Optimizer           : Adam")  # Prints the optimizer algorithm used to update the model parameters.
    print(f"Loss Function       : Cross Entropy")  # Prints the loss function used to measure next-character prediction error.
    print(f"Dropout Rate        : {DEFAULT_DROPOUT_RATE}")  # Prints the probability used by dropout to zero activations during training.
    print(f"Gradient Clipping   : 1.0")  # Prints the maximum gradient norm used by gradient clipping.
    print(f"Random Seed         : {SEED}")  # Prints the fixed seed used for repeatable random-number generation.
    print(f"Evaluation Interval : {args.eval_every}")  # Prints how frequently the model is evaluated during training.
    print(f"Evaluation Batches  : {args.eval_batches}")  # Prints how many batches are averaged for each evaluation.

    # 06 - Training performance
    print("\n---------- 06. TRAINING PERFORMANCE ----------")  # Prints the heading for the training-performance section.

    print(  # Begins a print call that formats the initial training loss.
        f"Initial Train Loss  : "  # Starts the text label for the initial training-loss value.
        f"{initial_losses['train']:.4f}"  # Reads the initial training loss and formats it to four decimal places.
    )  # Closes the print call for the initial training loss.

    print(  # Begins a print call that formats the initial validation loss.
        f"Initial Val Loss    : "  # Starts the text label for the initial validation-loss value.
        f"{initial_losses['val']:.4f}"  # Reads the initial validation loss and formats it to four decimal places.
    )  # Closes the print call for the initial validation loss.

    print(  # Begins a print call that formats the final training loss.
        f"Final Train Loss    : "  # Starts the text label for the final training-loss value.
        f"{final_losses['train']:.4f}"  # Reads the final training loss and formats it to four decimal places.
    )  # Closes the print call for the final training loss.

    print(  # Begins a print call that formats the final validation loss.
        f"Final Val Loss      : "  # Starts the text label for the final validation-loss value.
        f"{final_losses['val']:.4f}"  # Reads the final validation loss and formats it to four decimal places.
    )  # Closes the print call for the final validation loss.

    print(f"Training Duration   : {elapsed_time:.2f} seconds")  # Prints total elapsed runtime rounded to two decimal places.

    # 07 - Computational information
    print("\n---------- 07. COMPUTATIONAL DETAILS ----------")  # Prints the heading for the computational-details section.
    print(f"Execution Backend   : PyTorch")  # Prints the software framework used to run the model.
    print(f"Device              : {device}")  # Prints the device object used by the model.
    print(f"Parameter Data Type : {dtype}")  # Prints the data type used for the model parameters.
    print(f"GPU Acceleration    : {'Enabled' if device.type == 'cuda' else 'Disabled'}")  # Reports GPU acceleration as enabled only if the model is on a CUDA device.

    # Final separator
    print("\n" + "=" * 65)  # Prints a blank line followed by the report's bottom divider.
    print("                 END OF TECHNICAL REPORT")  # Prints the report-ending label.
    print("=" * 65 + "\n")  # Prints the final divider and a trailing blank line.

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
        description="Train a small PyTorch character-level GPT on Tiny Shakespeare, with validation."  # Provides the description displayed in the command-line help output.
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

    run_start_time = time.perf_counter() # This measures the complete run, including dataset loading, model initialization, training and generation.

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
        vocab_size,  # Passes the vocabulary size as the first positional constructor argument.
        blocks,  # Passes the number of Transformer blocks as the second constructor argument.
        dim_model,  # Passes the hidden embedding dimension as the third constructor argument.
        n_heads,  # Passes the number of attention heads as the fourth constructor argument.
        max_seq_len=max(5000, args.seq_len),  # Sets positional encoding capacity to at least 5,000 positions or the requested context length.
    ).to(device=DEVICE, dtype=MODEL_DTYPE)  # Places the model on CPU and preserves float64 precision
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE, weight_decay=0.01)  # Creates the PyTorch Adam optimizer
    mask = create_causal_mask(args.seq_len, device=DEVICE, dtype=MODEL_DTYPE)  # Builds the causal mask for training sequences

    print(f"Building ShaunGPT: {blocks} block(s), d_model={dim_model}, heads={n_heads}, context={args.seq_len}")  # Displays model architecture
    print(f"Training for {args.steps} steps...\n")  # Displays the planned number of training steps

    initial_losses = None  # Initializes the initial-loss variable. 
    final_losses = None  # Initializes the final-loss variable.
    # Evaluate the model before any training updates
    initial_losses = estimate_loss(
        model,
        train_data,
        val_data,
        args.batch_size,
        args.seq_len,
        mask,
        args.eval_batches
    )

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
                model,  # Passes the model instance to the loss-estimation function.
                train_data,  # Passes the training split so its loss can be estimated.
                val_data,  # Passes the validation split so its loss can be estimated.
                args.batch_size,  # Passes the batch size used to sample evaluation batches.
                args.seq_len,  # Passes the sequence length used for evaluation.
                mask,  # Passes the precomputed causal attention mask.
                args.eval_batches,  # Passes the number of batches to average for each split.
            )  # Completes the evaluation call
            current_train = losses["train"]  # Extracts the training loss from the evaluation results dictionary.
            current_val = losses["val"]  # Extracts the validation loss from the evaluation results dictionary.

            # Store the latest evaluation results
            final_losses = losses.copy()  # Copies the newest evaluation results as the latest/final loss measurements.
            
            print(f"  Evaluation | train loss: {current_train:.4f} | validation loss: {current_val:.4f}")  # Prints the current evaluation's training and validation losses to four decimal places.

        if args.sample_every and (step % args.sample_every == 0 or step == args.steps):  # Checks whether sampling is enabled and due
            prompt = "First Citizen:"  # Sets the text prompt used to start sample generation
            # Converts available prompt characters to token IDs
            if all(ch in char_to_int for ch in prompt):
                prompt_tokens = [char_to_int[ch] for ch in prompt]
            else:
                prompt_tokens = [int(data[0].item())]
            if not prompt_tokens:  # Checks whether the prompt produced any valid vocabulary tokens
                prompt_tokens = [int(data[0].item())]  # Uses the dataset's first token as a fallback
            generated_tokens = model.generate(  # Starts autoregressive generation from the prompt token IDs
                prompt_tokens,  # Passes the prompt token IDs as the first argument to the generation method.
                max_new_tokens=DEFAULT_GENERATE_TOKENS,  # Sets the number of newly generated tokens.
                seq_len=args.seq_len,  # Sets the context window used during generation.
                temperature=0.8,  # Sets the sampling temperature to 0.8.
                top_k=min(20, vocab_size),  # Limits sampling to at most 20 candidate tokens, or the vocabulary size if smaller.
            )  # Completes the text-generation call
            generated_text = "".join(int_to_char[token] for token in generated_tokens)  # Converts generated IDs back into characters
            print("\n--- ShaunGPT sample ---")  # Prints a heading before generated text
            print(generated_text)  # Displays the generated character sequence
            print("-----------------------\n")  # Prints a separator after the generated sample

    print("ShaunGPT Training successfully finished!")  # Prints a message confirming that the training loop has completed.

    # Calculate total execution time
    elapsed_time = time.perf_counter() - run_start_time  # Calculates elapsed wall-clock time using the high-resolution performance counter.

    # Identify dataset source
    data_path = (  # Begins a conditional expression that chooses the dataset's absolute path.
        Path(args.data).expanduser().resolve()  # Expands the user-home shortcut and resolves the supplied dataset path.
        if args.data  # Uses the supplied path when the --data argument was provided.
        else Path(__file__).resolve().with_name("input.txt")  # Otherwise resolves input.txt next to this Python script.
    )  # Closes the parenthesized conditional expression.

    # Generate technical report
    print_technical_report(  # Begins calling the report function with named arguments.
        model=model,  # Supplies the trained GPT model to the report function.
        args=args,  # Supplies the parsed command-line settings.
        data=data,  # Supplies the complete encoded dataset.
        train_data=train_data,  # Supplies the training portion of the dataset.
        val_data=val_data,  # Supplies the validation portion of the dataset.
        vocab_size=vocab_size,  # Supplies the vocabulary size.
        initial_losses=initial_losses,  # Supplies the initial evaluation-loss dictionary.
        final_losses=final_losses,  # Supplies the latest evaluation-loss dictionary.
        elapsed_time=elapsed_time,  # Supplies the measured elapsed runtime.
        data_path=data_path,  # Supplies the resolved dataset file path.
    )  # Closes the report-function call.

if __name__ == "__main__":  # Runs the following block only when this file is executed directly, not imported as a module.
    main()  # Calls main() to parse arguments, train the GPT, generate samples, and print the technical report.
