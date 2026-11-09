from pathlib import Path  # Imports the Path class from pathlib module for handling file system paths
import argparse          # Imports argparse module for parsing command-line arguments
import numpy as np       # Imports NumPy library and assigns it the standard alias np

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

# ========================= Utility functions =========================
def create_causal_mask(seq_len: int) -> np.ndarray:  # Defines a function to create a causal attention mask for autoregressive decoding
    """Return (seq_len, seq_len) additive mask: future positions are -inf."""  # Docstring explaining the function purpose
    if seq_len <= 0:  # Checks if the sequence length is less than or equal to zero
        raise ValueError("seq_len must be positive")  # Raises an error if sequence length is invalid
    return np.where(np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1), -np.inf, 0.0)  # Returns a matrix with -inf on and above the diagonal and 0.0 elsewhere

def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:  # Defines a function for numerically stable softmax computation
    """Numerically stable softmax."""  # Docstring explaining the function purpose
    shifted = logits - np.max(logits, axis=axis, keepdims=True)  # Subtracts the maximum logit value along the specified axis to prevent overflow
    exp_values = np.exp(shifted)  # Computes the exponential of the shifted logits
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)  # Divides exponential values by their sum along the axis to get probabilities

# ========================= Embedding layer =========================
class EmbeddingsLEGO:  # Defines a class for token embedding lookup and backpropagation
    def __init__(self, vocab_size: int, embedding_dim: int):  # Constructor method initializing embedding weights and gradients
        self.embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float64) * 0.02  # Initializes embedding matrix with small random normal values
        self.inputs = None  # Initializes a placeholder to store forward pass inputs
        self.grad_embeddings = np.zeros_like(self.embeddings)  # Initializes gradient storage for embeddings with zeros

    def forward(self, inputs: np.ndarray) -> np.ndarray:  # Forward pass method to convert token IDs to embedding vectors
        self.inputs = np.asarray(inputs, dtype=np.int64)  # Converts input tokens to a 64-bit integer numpy array and stores them
        return self.embeddings[self.inputs]  # Looks up and returns embedding vectors corresponding to the input token IDs

    def backward(self, grad_output: np.ndarray) -> None:  # Backward pass method to compute gradients for the embeddings
        self.grad_embeddings.fill(0.0)  # Resets the embedding gradients to zero before accumulation
        # Multiple occurrences of a character must accumulate their gradients.  # Comment explaining gradient accumulation for repeated tokens
        np.add.at(  # Uses numpy add.at to correctly accumulate gradients at duplicate indices
            self.grad_embeddings,  # Target array where gradients are added
            self.inputs.reshape(-1),  # Flattened input indices
            grad_output.reshape(-1, grad_output.shape[-1]),  # Flattened gradient outputs from the next layer
        )  # Closes the preceding function or method call

# ========================= Linear layer =========================
class LinearLEGO:  # Defines a fully connected linear transformation layer class
    def __init__(self, in_features: int, out_features: int, include_bias: bool = True):  # Constructor for weights, bias, and gradients
        self.W = np.random.randn(in_features, out_features).astype(np.float64) * 0.02  # Initializes weight matrix with small random values
        self.include_bias = include_bias  # Stores whether bias is included in this layer
        self.b = np.zeros((1, out_features), dtype=np.float64) if include_bias else None  # Initializes bias vector to zeros if enabled, else None
        self.inputs = None  # Placeholder for storing inputs during forward pass
        self.grad_W = np.zeros_like(self.W)  # Initializes weight gradients to zero
        self.grad_b = np.zeros_like(self.b) if include_bias else None  # Initializes bias gradients to zero if bias is included

    def forward(self, inputs: np.ndarray) -> np.ndarray:  # Forward pass for the linear layer
        self.inputs = inputs  # Stores input tensor for the backward pass
        out = inputs @ self.W  # Computes matrix multiplication of inputs and weights
        if self.include_bias:  # Checks if bias is included
            out = out + self.b  # Adds bias vector to the output matrix
        return out  # Returns the transformed output tensor

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass to compute gradients for weights, bias, and inputs
        inputs_flat = self.inputs.reshape(-1, self.inputs.shape[-1])  # Reshapes input tensor into a 2D matrix
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1])  # Reshapes gradient output into a 2D matrix
        self.grad_W = inputs_flat.T @ grad_flat  # Computes weight gradients via matrix multiplication
        if self.include_bias:  # Checks if bias is included
            self.grad_b = np.sum(grad_flat, axis=0, keepdims=True)  # Computes bias gradients by summing across the batch/sequence dimensions
        return grad_output @ self.W.T  # Computes and returns gradients with respect to the inputs

# ========================= ReLU activation =========================
class ReLU_LEGO:  # Defines the Rectified Linear Unit (ReLU) activation function class
    def __init__(self):  # Constructor for ReLU layer
        self.inputs = None  # Placeholder for storing inputs

    def forward(self, inputs: np.ndarray) -> np.ndarray:  # Forward pass applying ReLU activation
        self.inputs = inputs  # Stores inputs for backward pass
        return np.maximum(0.0, inputs)  # Returns element-wise maximum of 0 and input

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass for ReLU activation
        return grad_output * (self.inputs > 0.0)  # Multiplies gradient by indicator where inputs were positive

# ========================= Dropout layer =========================
class DropoutLEGO:  # Defines this function or class
    def __init__(self, dropout_rate=0.10):  # Defines this function or class
        if not 0.0 <= dropout_rate < 1.0:  # Checks this condition before executing the indented block
            raise ValueError("Dropout rate must be between 0 and 1.")  # Raises an exception to report an invalid condition

        self.dropout_rate = dropout_rate  # Stores the probability that each activation will be dropped during training
        self.mask = None  # Initializes the dropout mask; None means dropout is not currently active

    def forward(self, inputs, training=False):  # Defines the forward pass and controls dropout with the training flag
        if not training or self.dropout_rate == 0:  # Skips dropout during evaluation or when the dropout rate is zero
            self.mask = None  # Clears the mask because no activations will be dropped
            return inputs  # Returns the input unchanged when dropout is disabled

        keep_probability = 1.0 - self.dropout_rate  # Calculates the probability that an activation will be kept

        self.mask = (  # Begins constructing a random binary mask with the same shape as the input
            np.random.random(inputs.shape) < keep_probability  # Creates True entries for activations selected to be kept
        ).astype(inputs.dtype)  # Converts the Boolean mask to the input's numeric data type

        self.mask /= keep_probability  # Scales kept activations to preserve their expected magnitude

        return inputs * self.mask  # Applies the dropout mask to the input activations

    def backward(self, grad_output):  # Defines the gradient pass through the dropout layer
        if self.mask is None:  # Checks whether dropout was inactive during the forward pass
            return grad_output  # Passes gradients through unchanged when dropout was inactive

        return grad_output * self.mask  # Applies the same mask and scaling to gradients
    
# ========================= Layer normalization =========================
class LayerNormLEGO:  # Defines the Layer Normalization class
    def __init__(self, input_dim: int, epsilon: float = 1e-5):  # Constructor initializing normalization parameters
        self.input_dim = input_dim  # Stores the input feature dimension size
        self.epsilon = epsilon  # Small constant added to variance to prevent division by zero
        self.gain = np.ones((1, input_dim), dtype=np.float64)  # Initializes learnable scale parameter (gamma) to ones
        self.bias = np.zeros((1, input_dim), dtype=np.float64)  # Initializes learnable shift parameter (beta) to zeros
        self.grad_gain = np.zeros_like(self.gain)  # Initializes gradient for gain parameter
        self.grad_bias = np.zeros_like(self.bias)  # Initializes gradient for bias parameter
        self.normalized = None  # Placeholder for normalized intermediate values
        self.inv_std = None  # Placeholder for inverse standard deviation

    def forward(self, inputs: np.ndarray) -> np.ndarray:  # Forward pass for layer normalization
        self.inputs = inputs  # Stores input tensor
        mean = np.mean(inputs, axis=-1, keepdims=True)  # Computes mean across the last feature axis
        centered = inputs - mean  # Subtracts mean from inputs to center them
        variance = np.mean(centered * centered, axis=-1, keepdims=True)  # Computes variance across the last axis
        self.inv_std = 1.0 / np.sqrt(variance + self.epsilon)  # Computes inverse standard deviation with epsilon for numerical stability
        self.normalized = centered * self.inv_std  # Normalizes centered inputs to have unit variance and zero mean
        return self.normalized * self.gain + self.bias  # Scales and shifts normalized values using learnable gain and bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass for layer normalization
        # Parameters have shape (1, D); reduce over all batch/sequence axes.  # Comment on parameter gradient reduction axes
        sum_axes = tuple(range(grad_output.ndim - 1))  # Defines axes to sum over for parameter gradients
        self.grad_gain = np.sum(grad_output * self.normalized, axis=sum_axes).reshape(self.gain.shape)  # Computes gradient for gain parameter
        self.grad_bias = np.sum(grad_output, axis=sum_axes).reshape(self.bias.shape)  # Computes gradient for bias parameter

        dim = grad_output.shape[-1]  # Gets the feature dimension size
        grad_normalized = grad_output * self.gain  # Scales gradient output by gain parameter
        sum_grad = np.sum(grad_normalized, axis=-1, keepdims=True)  # Sums scaled gradients across the feature axis
        sum_grad_times_xhat = np.sum(grad_normalized * self.normalized, axis=-1, keepdims=True)  # Sums product of scaled gradients and normalized inputs
        grad_input = (self.inv_std / dim) * (  # Computes gradient with respect to inputs using analytic derivative formula
            dim * grad_normalized - sum_grad - self.normalized * sum_grad_times_xhat  # Core gradient calculation terms
        )  # Closing parenthesis for gradient formula expression
        return grad_input  # Returns computed input gradients

# ========================= Sinusoidal position encodings =========================
class PositionalEncodingsLEGO:  # Defines the positional encoding module class
    def __init__(self, dim_model: int, max_seq_len: int = 5000):  # Constructor for sinusoidal positional encodings
        if dim_model <= 0 or max_seq_len <= 0:  # Validates model dimension and maximum sequence length
            raise ValueError("dim_model and max_seq_len must be positive")  # Raises error if parameters are invalid
        self.dim_model = dim_model  # Stores model dimension
        self.max_seq_len = max_seq_len  # Stores maximum sequence length
        pe = np.zeros((max_seq_len, dim_model), dtype=np.float64)  # Initializes positional encoding matrix with zeros
        position = np.arange(max_seq_len, dtype=np.float64)[:, None]  # Creates position index array
        div_term = np.exp(np.arange(0, dim_model, 2) * (-np.log(10000.0) / dim_model))  # Computes logarithmic division terms for frequencies
        pe[:, 0::2] = np.sin(position * div_term)  # Applies sine function to even indices of positional encodings
        # Handles both even and odd model dimensions safely.  # Comment on handling odd/even dimensions
        pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])  # Applies cosine function to odd indices of positional encodings
        self.pe = pe[None, :, :]  # Adds batch dimension to positional encoding matrix

    def forward(self, inputs: np.ndarray) -> np.ndarray:  # Forward pass adding positional encodings to inputs
        seq_len = inputs.shape[1]  # Gets sequence length from input tensor shape
        if seq_len > self.max_seq_len:  # Checks if sequence length exceeds maximum allowed length
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")  # Raises error if sequence is too long
        return inputs + self.pe[:, :seq_len, :]  # Adds sliced positional encodings to input tensor and returns it

# ========================= Softmax layer (used by attention) =========================
class SoftmaxLEGO:  # Defines a dedicated Softmax layer class for attention mechanism
    def __init__(self):  # Constructor for Softmax layer
        self.softmax_out = None  # Placeholder for storing softmax output

    def forward(self, inputs: np.ndarray) -> np.ndarray:  # Forward pass applying stable softmax
        self.softmax_out = stable_softmax(inputs, axis=-1)  # Computes and stores stable softmax outputs
        return self.softmax_out  # Returns softmax outputs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass computing Jacobian-vector product for softmax
        # Jacobian-vector product without explicitly constructing the Jacobian.  # Comment explaining efficient gradient calculation
        dot = np.sum(grad_output * self.softmax_out, axis=-1, keepdims=True)  # Computes dot product of gradients and softmax outputs
        return self.softmax_out * (grad_output - dot)  # Returns computed gradients with respect to softmax inputs

# ========================= Multi-head causal self-attention =========================
class MultiHeadAttentionLEGO:  # Defines the Multi-Head Causal Self-Attention layer class
    def __init__(self, dim_in: int, dim_out: int, n_heads: int):  # Constructor initializing attention parameters and projections
        if n_heads <= 0 or dim_out % n_heads != 0:  # Validates that head count and output dimensions are compatible
            raise ValueError("dim_out must be divisible by n_heads")  # Raises error if dimensions don't match head count
        self.dim_in = dim_in  # Stores input dimension
        self.dim_out = dim_out  # Stores output dimension
        self.n_heads = n_heads  # Stores number of attention heads
        self.head_dim = dim_out // n_heads  # Computes dimension size per individual attention head

        self.query_linear = LinearLEGO(dim_in, dim_out, include_bias=False)  # Linear projection layer for queries
        self.key_linear = LinearLEGO(dim_in, dim_out, include_bias=False)    # Linear projection layer for keys
        self.value_linear = LinearLEGO(dim_in, dim_out, include_bias=False)  # Linear projection layer for values
        self.final_projection = LinearLEGO(dim_out, dim_out, include_bias=False)  # Final output linear projection layer
        self.softmax = SoftmaxLEGO()  # Softmax layer for attention weights

        self.queries = None  # Placeholder for query tensors
        self.keys = None     # Placeholder for key tensors
        self.values = None   # Placeholder for value tensors
        self.attention_weights = None  # Placeholder for computed attention weights

    def forward(self, inputs: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:  # Forward pass for multi-head attention
        batch, time_steps, _ = inputs.shape  # Unpacks batch size and time steps from input shape
        queries = self.query_linear.forward(inputs).reshape(batch, time_steps, self.n_heads, self.head_dim)  # Projects and reshapes queries for multiple heads
        keys = self.key_linear.forward(inputs).reshape(batch, time_steps, self.n_heads, self.head_dim)        # Projects and reshapes keys for multiple heads
        values = self.value_linear.forward(inputs).reshape(batch, time_steps, self.n_heads, self.head_dim)    # Projects and reshapes values for multiple heads

        self.queries = queries.transpose(0, 2, 1, 3)  # Transposes queries to shape (batch, n_heads, time_steps, head_dim)
        self.keys = keys.transpose(0, 2, 1, 3)        # Transposes keys to shape (batch, n_heads, time_steps, head_dim)
        self.values = values.transpose(0, 2, 1, 3)    # Transposes values to shape (batch, n_heads, time_steps, head_dim)

        scores = (self.queries @ self.keys.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)  # Computes scaled dot-product attention scores
        if mask is not None:  # Checks if a causal attention mask is provided
            scores = scores + mask[None, None, :, :]  # Adds causal mask to attention scores to block future tokens
        self.attention_weights = self.softmax.forward(scores)  # Applies softmax to get normalized attention probabilities

        attended = self.attention_weights @ self.values  # Computes weighted sum of values using attention weights
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, time_steps, self.dim_out)  # Transposes and flattens head dimensions back to model dimension
        return self.final_projection.forward(attended)  # Passes attended outputs through the final linear projection layer

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass for multi-head attention
        grad_projected = self.final_projection.backward(grad_output)  # Computes backward pass through final projection
        batch, time_steps, _ = grad_projected.shape  # Gets batch size and time steps from gradient shape
        grad_heads = grad_projected.reshape(batch, time_steps, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)  # Reshapes and transposes gradients for attention heads

        grad_attention_weights = grad_heads @ self.values.transpose(0, 1, 3, 2)  # Computes gradients with respect to attention weights
        grad_values = self.attention_weights.transpose(0, 1, 3, 2) @ grad_heads   # Computes gradients with respect to values

        grad_scores = self.softmax.backward(grad_attention_weights) / np.sqrt(self.head_dim)  # Backpropagates through softmax and scales by square root of head dimension
        grad_queries = grad_scores @ self.keys  # Computes gradients with respect to queries
        grad_keys = grad_scores.transpose(0, 1, 3, 2) @ self.queries  # Computes gradients with respect to keys

        grad_queries = grad_queries.transpose(0, 2, 1, 3).reshape(batch, time_steps, self.dim_out)  # Reshapes query gradients back to model shape
        grad_keys = grad_keys.transpose(0, 2, 1, 3).reshape(batch, time_steps, self.dim_out)        # Reshapes key gradients back to model shape
        grad_values = grad_values.transpose(0, 2, 1, 3).reshape(batch, time_steps, self.dim_out)    # Reshapes value gradients back to model shape

        return (  # Combines and returns gradients from query, key, and value linear layers
            self.query_linear.backward(grad_queries)  # Backpropagates through query linear layer
            + self.key_linear.backward(grad_keys)     # Backpropagates through key linear layer
            + self.value_linear.backward(grad_values) # Backpropagates through value linear layer
        )  # Closing parenthesis for return statement

# ========================= Feed-forward network =========================
class FeedForwardNetworkLEGO:  # Defines the Position-wise Feed-Forward Network class
    def __init__(self, input_units: int, hidden_units: int, output_units: int, dropout_rate=0.10):  # Constructor initializing linear layers and activation
        self.linear1 = LinearLEGO(input_units, hidden_units)  # First linear expansion layer
        self.activation = ReLU_LEGO()                       # ReLU activation function layer
        self.dropout = DropoutLEGO(dropout_rate)            # Dropouts layer
        self.linear2 = LinearLEGO(hidden_units, output_units)  # Second linear projection layer back to model dimension

    def forward(self, inputs, training=False):  # Defines the forward pass and controls dropout with the training flag
        x = self.linear1.forward(inputs)  # Projects each input feature vector into the larger hidden dimension
        x = self.activation.forward(x)  # Applies the ReLU nonlinearity to the hidden activations
        x = self.dropout.forward(x, training=training)  # Applies dropout to hidden activations when training
        return self.linear2.forward(x)  # Projects hidden activations back to the model dimension

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass through feed-forward network
        grad = self.linear2.backward(grad_output)  # Backpropagates through second linear layer
        grad = self.dropout.backward(grad)         # Backpropagates through dropout function
        grad = self.activation.backward(grad)      # Backpropagates through ReLU activation function
        return self.linear1.backward(grad)         # Backpropagates through first linear layer and returns input gradients

# ========================= Transformer decoder block =========================
class TransformerBlockLEGO:  # Defines a single Transformer Decoder Block class
    """Pre-layer-normalized transformer block with residual connections."""  # Docstring describing the transformer block architecture
    def __init__(self, dim_model: int, n_heads: int):  # Constructor initializing attention, layer norms, and feed-forward network
        self.multi_head_attention = MultiHeadAttentionLEGO(dim_model, dim_model, n_heads)  # Multi-head attention sub-layer
        self.layernorm1 = LayerNormLEGO(dim_model)  # First layer normalization applied before attention (Pre-LN)
        self.ffn = FeedForwardNetworkLEGO(  # Stores or updates this object's state
            dim_model,  # Performs this step in the model's computation or training workflow
            4 * dim_model,  # Performs this step in the model's computation or training workflow
            dim_model,  # Performs this step in the model's computation or training workflow
            dropout_rate=DEFAULT_DROPOUT_RATE  # Performs this step in the model's computation or training workflow
        )  # Feed-forward network sub-layer
        self.layernorm2 = LayerNormLEGO(dim_model)  # Second layer normalization applied before FFN (Pre-LN)

    def forward(self, inputs: np.ndarray, mask_attention: np.ndarray | None = None, training=False) -> np.ndarray:  # Forward pass for transformer block
        self.inputs = inputs  # Stores input tensor
        normalized1 = self.layernorm1.forward(inputs)  # Applies first layer normalization
        attention_output = self.multi_head_attention.forward(normalized1, mask=mask_attention)  # Applies multi-head attention on normalized input
        self.mid = inputs + attention_output  # Adds residual connection around attention sub-layer
        normalized2 = self.layernorm2.forward(self.mid)  # Applies second layer normalization
        ffn_output = self.ffn.forward(
            normalized2,
            training=training
        )  # Applies feed-forward network on second normalized tensor
        return self.mid + ffn_output  # Adds residual connection around feed-forward sub-layer and returns output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:  # Backward pass for transformer block
        # Residual path around the feed-forward network.  # Comment on FFN residual gradient path
        grad_mid = grad_output + self.layernorm2.backward(self.ffn.backward(grad_output))  # Computes gradients through FFN block and residual addition
        # Residual path around attention.  # Comment on attention residual gradient path
        grad_inputs = grad_mid + self.layernorm1.backward(self.multi_head_attention.backward(grad_mid))  # Computes gradients through attention block and residual addition
        return grad_inputs  # Returns input gradients for the transformer block

    def get_params_and_grads(self):  # Helper method to collect all learnable parameters and their corresponding gradients
        result = []  # Initializes list to store parameter-gradient tuples
        result.extend([  # Extends list with tuples of (layer, param_name, grad_name) for optimizer
            (self.multi_head_attention.query_linear, "W", "grad_W"),  # Query weights and gradients
            (self.multi_head_attention.key_linear, "W", "grad_W"),    # Key weights and gradients
            (self.multi_head_attention.value_linear, "W", "grad_W"),  # Value weights and gradients
            (self.multi_head_attention.final_projection, "W", "grad_W"),  # Attention projection weights and gradients
            (self.layernorm1, "gain", "grad_gain"),  # First layer norm gain parameter and gradient
            (self.layernorm1, "bias", "grad_bias"),  # First layer norm bias parameter and gradient
            (self.ffn.linear1, "W", "grad_W"),       # FFN first linear weights and gradients
            (self.ffn.linear1, "b", "grad_b"),       # FFN first linear bias and gradients
            (self.ffn.linear2, "W", "grad_W"),       # FFN second linear weights and gradients
            (self.ffn.linear2, "b", "grad_b"),       # FFN second linear bias and gradients
            (self.layernorm2, "gain", "grad_gain"),  # Second layer norm gain parameter and gradient
            (self.layernorm2, "bias", "grad_bias"),  # Second layer norm bias parameter and gradient
        ])  # Closing bracket for list extension
        return result  # Returns the collected parameters and gradients list

# ========================= Cross-entropy loss =========================
class CrossEntropyWithSoftmaxLEGO:  # Defines Combined Softmax and Cross-Entropy Loss class
    def __init__(self):  # Constructor initializing internal state variables
        self.probabilities = None  # Placeholder for computed probabilities
        self.labels_flat = None     # Placeholder for flattened target labels
        self.batch_time = None      # Placeholder for total number of tokens in batch
        self.logits_shape = None    # Placeholder for original logits tensor shape

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:  # Forward pass computing cross-entropy loss
        if logits.ndim != 3:  # Validates that logits have 3 dimensions (batch, time, vocab)
            raise ValueError("logits must have shape (batch, time, vocabulary)")  # Raises error if logit shape is incorrect
        batch, time_steps, vocab_size = logits.shape  # Unpacks shape dimensions of logits tensor
        labels_flat = np.asarray(labels, dtype=np.int64).reshape(-1)  # Flattens and converts labels to 64-bit integer numpy array
        if labels_flat.size != batch * time_steps:  # Validates label size against logits dimensions
            raise ValueError("labels do not match logits batch/time dimensions")  # Raises error if sizes don't match
        if np.any(labels_flat < 0) or np.any(labels_flat >= vocab_size):  # Checks for out-of-bounds label indices
            raise ValueError("label index is outside the vocabulary")  # Raises error if any label is invalid

        flat_logits = logits.reshape(-1, vocab_size)  # Reshapes logits into a 2D matrix of shape (batch * time, vocab_size)
        max_logits = np.max(flat_logits, axis=-1, keepdims=True)  # Computes maximum logit per token for numerical stability
        exp_logits = np.exp(flat_logits - max_logits)  # Computes exponentiated shifted logits
        exp_sum = np.sum(exp_logits, axis=-1, keepdims=True)  # Computes sum of exponentiated logits for normalization partition function
        log_partition = max_logits + np.log(exp_sum)  # Computes log partition function (log-sum-exp)
        log_probs = flat_logits - log_partition  # Computes log probabilities for all classes
        loss = -np.mean(log_probs[np.arange(labels_flat.size), labels_flat])  # Computes mean negative log-likelihood loss for correct labels

        self.probabilities = exp_logits / exp_sum  # Stores computed softmax probabilities for backward pass
        self.labels_flat = labels_flat             # Stores flattened labels for backward pass
        self.batch_time = labels_flat.size         # Stores total batch-time token count
        self.logits_shape = logits.shape           # Stores original logits shape
        return float(loss)                         # Returns scalar cross-entropy loss value

    def backward(self) -> np.ndarray:  # Backward pass computing gradient of loss with respect to logits
        """Gradient of mean cross entropy with respect to the original logits."""  # Docstring explaining backward pass method
        if self.probabilities is None:  # Checks if forward pass was called
            raise RuntimeError("Call forward() before backward().")  # Raises error if forward was not run first
        grad = self.probabilities.copy()  # Copies stored softmax probabilities
        grad[np.arange(self.labels_flat.size), self.labels_flat] -= 1.0  # Subtracts 1 from the probability of the true target class
        grad /= self.batch_time  # Normalizes gradients by total number of tokens
        return grad.reshape(self.logits_shape)  # Reshapes and returns gradients back to original logit tensor shape

# ========================= Adam optimizer =========================
class AdamOptimizerLEGO:  # Defines the Adam Optimizer class with gradient clipping
    """Adam with global gradient clipping and parameter references."""  # Docstring describing optimizer features
    def __init__(self, model_params_list, lr: float = 0.001, beta1: float = 0.9,  # Constructor initializing Adam hyperparameters and states
                 beta2: float = 0.999, epsilon: float = 1e-8, grad_clip_norm: float = 1.0):  # Additional Adam parameters
        self.params = list(model_params_list)  # Stores list of model parameters and gradients
        self.lr = lr                     # Stores learning rate
        self.beta1 = beta1               # Stores exponential decay rate for first moment estimates
        self.beta2 = beta2               # Stores exponential decay rate for second moment estimates
        self.epsilon = epsilon           # Small constant for numerical stability in denominator
        self.grad_clip_norm = grad_clip_norm  # Maximum allowed norm for gradient clipping
        self.t = 0                       # Initializes timestep counter
        self.m = [np.zeros_like(getattr(layer, name)) for layer, name, _ in self.params]  # Initializes first moment vector list with zeros
        self.v = [np.zeros_like(getattr(layer, name)) for layer, name, _ in self.params]  # Initializes second moment vector list with zeros

    def step(self, model_params_list=None) -> None:  # Performs a single optimization step (parameter update)
        params = self.params if model_params_list is None else list(model_params_list)  # Selects parameter list
        if len(params) != len(self.m):  # Validates that parameter list length hasn't changed
            raise ValueError("The parameter list changed after Adam was initialized")  # Raises error if parameter count changed

        grads = [getattr(layer, grad_name) for layer, _, grad_name in params]  # Collects current gradients for all parameters
        global_norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))  # Computes global L2 norm of all gradients combined
        clip_scale = min(1.0, self.grad_clip_norm / (global_norm + 1e-12))  # Computes scaling factor for gradient clipping

        self.t += 1  # Increments timestep counter
        for i, (layer, param_name, grad_name) in enumerate(params):  # Iterates through each parameter layer
            param = getattr(layer, param_name)  # Retrieves current parameter weights
            grad = getattr(layer, grad_name) * clip_scale  # Retrieves and clips gradient using scaling factor
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad  # Updates biased first moment estimate (mean)
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)  # Updates biased second moment estimate (uncentered variance)
            m_hat = self.m[i] / (1.0 - self.beta1 ** self.t)  # Computes bias-corrected first moment estimate
            v_hat = self.v[i] / (1.0 - self.beta2 ** self.t)  # Computes bias-corrected second moment estimate
            updated = param - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)  # Computes updated parameter values using Adam formula
            setattr(layer, param_name, updated)  # Assigns updated weights back to the layer parameter attribute

# ========================= GPT model =========================
class GPTLEGO:  # Defines the complete Character-level GPT Model class
    def __init__(self, vocab_size: int, blocks: int, dim_model: int, n_heads: int,  # Constructor initializing GPT architecture layers
                 max_seq_len: int = 5000):  # Maximum sequence length parameter
        if vocab_size <= 1:  # Validates vocabulary size
            raise ValueError("vocab_size must be greater than 1")  # Raises error if vocabulary is too small
        if blocks <= 0:  # Validates block count
            raise ValueError("blocks must be positive")  # Raises error if block count is invalid
        if dim_model <= 0 or dim_model % n_heads != 0:  # Validates model dimension and heads divisibility
            raise ValueError("dim_model must be positive and divisible by n_heads")  # Raises error if dimensions are invalid

        self.vocab_size = vocab_size  # Stores vocabulary size
        self.embedding_decoder = EmbeddingsLEGO(vocab_size, dim_model)  # Initializes token embedding layer
        self.positional_encodings_decoder = PositionalEncodingsLEGO(dim_model, max_seq_len)  # Initializes positional encoding layer
        self.decoder_blocks = [TransformerBlockLEGO(dim_model, n_heads) for _ in range(blocks)]  # Creates list of transformer decoder blocks
        self.final_layernorm = LayerNormLEGO(dim_model)  # Final layer normalization before output projection
        self.final_linear = LinearLEGO(dim_model, vocab_size)  # Final linear projection layer mapping hidden state to vocabulary logits
        self.loss_function = CrossEntropyWithSoftmaxLEGO()  # Loss function instance

    def forward(self, input_tokens_decoder: np.ndarray, mask_attention: np.ndarray | None = None, training=False) -> np.ndarray:  # Forward pass for full GPT model
        x = self.embedding_decoder.forward(input_tokens_decoder)  # Converts token IDs to embedding vectors
        x = self.positional_encodings_decoder.forward(x)          # Adds positional encodings to embeddings
        for decoder in self.decoder_blocks: # Iterates through each transformer block
            x = decoder.forward(
                x,
                mask_attention=mask_attention, # Passes hidden states through transformer block with attention mask
                training=training
            )
        x = self.final_layernorm.forward(x)                      # Applies final layer normalization
        return self.final_linear.forward(x)                      # Projects hidden states to vocabulary logits and returns them

    def backward(self, grad_output: np.ndarray) -> None:  # Backward pass for full GPT model
        grad_hidden = self.final_linear.backward(grad_output)      # Backpropagates through final linear output projection
        grad_hidden = self.final_layernorm.backward(grad_hidden)   # Backpropagates through final layer normalization
        for decoder in reversed(self.decoder_blocks):             # Iterates backwards through transformer decoder blocks
            grad_hidden = decoder.backward(grad_hidden)           # Backpropagates through each transformer block
        self.embedding_decoder.backward(grad_hidden)              # Backpropagates through token embedding layer

    def get_params_and_grads(self):  # Collects all parameters and gradients across the entire GPT model
        params = [(self.embedding_decoder, "embeddings", "grad_embeddings")]  # Starts with embedding parameters
        for block in self.decoder_blocks:  # Iterates through transformer blocks
            params.extend(block.get_params_and_grads())  # Adds parameters and gradients from each block
        params.extend([  # Adds final layer norm and final linear projection parameters and gradients
            (self.final_layernorm, "gain", "grad_gain"),  # Final layer norm gain parameters and gradients
            (self.final_layernorm, "bias", "grad_bias"),  # Final layer norm bias parameters and gradients
            (self.final_linear, "W", "grad_W"),           # Final linear weights and gradients
            (self.final_linear, "b", "grad_b"),           # Final linear bias and gradients
        ])  # Closing bracket for list extension
        return params  # Returns complete list of all model parameters and gradients

    def generate(self, start_tokens, max_new_tokens: int, seq_len: int,  # Autoregressive text generation method
                 temperature: float = 1.0, top_k: int | None = None) -> list[int]:  # Generation parameters (temperature and top-k filtering)
        """Generate max_new_tokens autoregressively; start_tokens are character IDs."""  # Docstring explaining text generation
        if temperature <= 0:  # Validates temperature
            raise ValueError("temperature must be greater than zero")  # Raises error if temperature is non-positive
        if seq_len <= 0 or max_new_tokens < 0:  # Validates sequence length and new token count
            raise ValueError("seq_len must be positive and max_new_tokens cannot be negative")  # Raises error if invalid

        generated = list(map(int, start_tokens))  # Initializes generated token list with starting token IDs
        if not generated:  # Checks if starting tokens are empty
            raise ValueError("start_tokens must contain at least one token")  # Raises error if no start tokens provided

        for _ in range(max_new_tokens):  # Loops to generate new tokens one by one autoregressively
            context = generated[-seq_len:]  # Extracts recent context slice limited by maximum sequence length
            x_input = np.asarray([context], dtype=np.int64)  # Converts context tokens to numpy array batch
            mask = create_causal_mask(len(context))          # Creates causal attention mask for current context length
            logits = self.forward(x_input, mask_attention=mask)  # Computes model forward pass to get logits for current context
            next_logits = logits[0, -1, :] / temperature        # Extracts logits for the last token position and scales by temperature

            if top_k is not None:  # Checks if top-k sampling filtering is enabled
                if top_k <= 0:     # Validates top_k value
                    raise ValueError("top_k must be positive when provided")  # Raises error if top_k is invalid
                k = min(top_k, next_logits.size)  # Ensures k does not exceed vocabulary size
                keep_indices = np.argpartition(next_logits, -k)[-k:]  # Finds indices of top-k highest logit values
                filtered = np.full_like(next_logits, -np.inf)        # Initializes filtered logits array with negative infinity
                filtered[keep_indices] = next_logits[keep_indices]   # Retains top-k logits and sets others to -inf
                next_logits = filtered  # Updates next_logits with filtered array

            probabilities = stable_softmax(next_logits)  # Computes stable probability distribution over vocabulary
            next_token = int(np.random.choice(self.vocab_size, p=probabilities))  # Samples next token ID from probability distribution
            generated.append(next_token)  # Appends newly generated token to the sequence list

        return generated  # Returns full list of generated token IDs

# ========================= Dataset loading =========================
def load_shakespeare(data_path: str | Path | None = None, max_chars: int | None = MAX_CHARS):  # Defines dataset loading utility function
    """Read the uploaded/local input.txt; no network download is required."""  # Docstring for dataset loading
    if data_path is None:  # Checks if data path is not specified
        data_path = Path(__file__).resolve().with_name("input.txt")  # Defaults to input.txt located next to script
    data_path = Path(data_path).expanduser().resolve()  # Resolves absolute path to dataset file
    if not data_path.is_file():  # Checks if dataset file exists
        raise FileNotFoundError(  # Raises file not found error with helpful troubleshooting message
            f"Could not find dataset: {data_path}\n"  # Performs this step in the model's computation or training workflow
            "Place input.txt in the same folder as Shaun_fixed.py, or pass --data PATH."  # Performs this step in the model's computation or training workflow
        )  # Closes the preceding function or method call

    text = data_path.read_text(encoding="utf-8")  # Reads entire text file content using UTF-8 encoding
    if not text:  # Checks if text file is empty
        raise ValueError(f"Dataset is empty: {data_path}")  # Raises error if dataset has no characters
    if max_chars is not None:  # Checks if maximum character limit is set
        if max_chars <= 0:  # Validates max_chars value
            raise ValueError("max_chars must be positive or None")  # Raises error if invalid
        text = text[:max_chars]  # Truncates text to maximum specified character limit

    chars = sorted(set(text))  # Extracts sorted list of unique characters in the dataset (vocabulary characters)
    char_to_int = {ch: i for i, ch in enumerate(chars)}  # Creates mapping dictionary from character to integer ID
    int_to_char = {i: ch for ch, i in char_to_int.items()}  # Creates mapping dictionary from integer ID to character
    encoded_data = np.asarray([char_to_int[ch] for ch in text], dtype=np.int64)  # Encodes text into 64-bit integer numpy array of token IDs
    if len(encoded_data) < 2:  # Validates encoded dataset length
        raise ValueError("Dataset needs at least two characters to create input/target pairs")  # Raises error if dataset is too short
    return encoded_data, len(chars), char_to_int, int_to_char, text  # Returns encoded data, vocab size, mappings, and raw text

def get_batch(data: np.ndarray, batch_size: int, seq_len: int):  # Defines function to sample random training batches
    """Sample random contiguous input/next-character target pairs."""  # Docstring for batch sampling function
    if len(data) <= seq_len:  # Checks if dataset is shorter than sequence length
        raise ValueError(f"Dataset must contain more than seq_len={seq_len} characters")  # Raises error if dataset is too short
    offsets = np.random.randint(0, len(data) - seq_len, size=batch_size)  # Samples random starting offsets for batch sequences
    x = np.stack([data[i:i + seq_len] for i in offsets])          # Creates input batch sequences tensor of length seq_len
    y = np.stack([data[i + 1:i + seq_len + 1] for i in offsets])  # Creates target batch sequences tensor shifted by 1 position
    return x, y  # Returns input batch x and target batch y

# ========================= Validation and checkpointing =========================
def estimate_loss(model, train_data, val_data, batch_size, seq_len, mask,  # Defines this function or class
                  eval_batches=10):  # Begins the indented block for this statement
    """Estimate train/validation loss on fresh batches without updating weights."""  # Performs this step in the model's computation or training workflow
    results = {}  # Creates a dictionary to hold the estimated loss for each dataset split
    for split_name, split_data in (("train", train_data), ("val", val_data)):  # Loops over the training and validation datasets separately
        losses = []  # Creates a list to collect loss values from evaluation batches
        for _ in range(eval_batches):  # Repeats evaluation over the requested number of batches
            x_batch, y_batch = get_batch(split_data, batch_size, seq_len)  # Samples input sequences and their next-character targets
            logits = model.forward(  # Runs the model to obtain vocabulary scores for every sequence position
                x_batch,  # Supplies the sampled input sequences to the model
                mask_attention=mask,  # Supplies the causal mask so positions cannot attend to future tokens
                training=False # Disables dropout during evaluation
            )  # Closes the text-generation call
            losses.append(model.loss_function.forward(logits, y_batch))  # Performs this step in the model's computation or training workflow
        results[split_name] = float(np.mean(losses))  # Stores the average loss for the current dataset split
    return results  # Returns the training and validation loss estimates

def main():  # Defines the main function that configures and runs model training
    parser = argparse.ArgumentParser(  # Creates a command-line argument parser for training options
        description="Train a small NumPy character-level GPT on Tiny Shakespeare, with validation."  # Performs this step in the model's computation or training workflow
    )  # Closes the argument parser constructor call
    parser.add_argument("--data", type=Path, default=None,  # Adds an optional command-line argument for the dataset file path
                        help="Path to input.txt (default: beside this script)")  # Explains the dataset path argument in command-line help
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,  # Adds the number of optimizer steps as a command-line option
                        help="Number of training steps")  # Explains the training-step argument
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,  # Adds the number of sequences per training batch as an option
                        help="Sequences per training batch")  # Explains the batch-size argument
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,  # Adds the context length as a command-line option
                        help="Context length")  # Explains the sequence-length argument
    parser.add_argument("--sample-every", type=int, default=DEFAULT_SAMPLE_EVERY,  # Adds an option controlling how often generated samples are printed
                        help="Generate text every N steps; 0 disables samples")  # Explains the sample interval and how to disable sampling
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,  # Adds an option limiting how many dataset characters are loaded
                        help="Maximum dataset characters; use 0 for the entire file")  # Explains the dataset-size limit argument
    parser.add_argument("--val-fraction", type=float, default=0.10,  # Adds the fraction of the dataset reserved for validation
                        help="Fraction of the final dataset reserved for validation (default: 0.10)")  # Explains the validation split argument
    parser.add_argument("--eval-every", type=int, default=250,  # Adds the interval between loss evaluations
                        help="Evaluate train/validation loss every N steps")  # Explains the evaluation interval
    parser.add_argument("--eval-batches", type=int, default=10,  # Adds the number of batches averaged during each evaluation
                        help="Number of batches averaged for each loss evaluation")  # Explains the evaluation batch-count argument
    args = parser.parse_args()  # Reads the command-line arguments supplied when the script is launched

    if args.steps <= 0 or args.batch_size <= 0 or args.seq_len <= 0:  # Checks that the main training dimensions and step count are positive
        parser.error("--steps, --batch-size and --seq-len must all be positive")  # Stops execution with a clear command-line usage error if a value is invalid
    if args.sample_every < 0:  # Checks that the sampling interval is not negative
        parser.error("--sample-every cannot be negative")  # Reports an invalid sampling interval and exits argument parsing
    if args.eval_every <= 0 or args.eval_batches <= 0:  # Checks that evaluation frequency and evaluation batch count are positive
        parser.error("--eval-every and --eval-batches must be positive")  # Reports invalid evaluation settings and exits argument parsing
    if not 0.0 < args.val_fraction < 0.5:  # Ensures the validation fraction is greater than zero and less than one half
        parser.error("--val-fraction must be between 0 and 0.5")  # Reports an invalid validation fraction and exits argument parsing

    np.random.seed(SEED)  # Seeds NumPy's random-number generator for repeatable initialization and sampling
    max_chars = None if args.max_chars == 0 else args.max_chars  # Uses the full file when the limit is zero; otherwise keeps the requested limit
    data, vocab_size, char_to_int, int_to_char, source_text = load_shakespeare(args.data, max_chars)  # Loads the text, encodes characters, and obtains vocabulary mappings
    split_index = int(len(data) * (1.0 - args.val_fraction))  # Calculates the index where the training portion ends
    train_data = data[:split_index]  # Assigns the earlier portion of encoded text to training
    val_data = data[split_index:]  # Assigns the final portion of encoded text to validation
    if len(train_data) <= args.seq_len or len(val_data) <= args.seq_len:  # Checks that both dataset splits can provide a full sequence and next-token target
        parser.error("Training and validation splits must each be longer than --seq-len")  # Stops with an error if either split is too short

    print(f"Dataset: {len(data):,} characters | vocabulary: {vocab_size} unique characters")  # Prints the dataset size and number of unique character tokens
    print(f"Source: {Path(args.data).resolve() if args.data else Path(__file__).resolve().with_name('input.txt')}")  # Prints the resolved path of the dataset file
    print(f"Split: {len(train_data):,} train characters ({1-args.val_fraction:.0%}) | "  # Begins printing the sizes and percentages of the training and validation splits
          f"{len(val_data):,} validation characters ({args.val_fraction:.0%})")  # Completes the dataset split summary

    dim_model = DEFAULT_DIM_MODEL  # Selects the configured transformer hidden dimension
    n_heads = DEFAULT_N_HEADS  # Selects the configured number of attention heads
    blocks = DEFAULT_BLOCKS  # Selects the configured number of transformer decoder blocks
    model = GPTLEGO(vocab_size, blocks, dim_model, n_heads,  # Constructs the character-level GPT using the selected architecture settings
                    max_seq_len=max(5000, args.seq_len))  # Sets the model's positional-encoding capacity to at least 5000 positions
    optimizer = AdamOptimizerLEGO(model.get_params_and_grads(), lr=DEFAULT_LEARNING_RATE)  # Creates Adam and connects it to all model parameters and gradients
    mask = create_causal_mask(args.seq_len)  # Builds the causal mask used for every training sequence

    print(f"Building ShaunGPT: {blocks} block(s), d_model={dim_model}, heads={n_heads}, context={args.seq_len}")  # Displays the model architecture configuration
    print(f"Training for {args.steps} steps...\n")  # Displays the planned number of training steps

    for step in range(1, args.steps + 1):  # Runs the training loop from step one through the requested final step
        x_batch, y_batch = get_batch(train_data, args.batch_size, args.seq_len)  # Samples a batch of training inputs and one-character-shifted targets
        logits = model.forward(
            x_batch,
            mask_attention=mask,
            training=True
        )  # Computes the model's vocabulary scores for each input position
        train_loss = model.loss_function.forward(logits, y_batch)  # Calculates the average next-character prediction loss
        grad_logits = model.loss_function.backward()  # Computes the gradient of the loss with respect to the model logits
        model.backward(grad_logits)  # Backpropagates the loss gradient through the complete GPT model
        optimizer.step()  # Updates all model parameters using Adam

        if step == 1 or step % 50 == 0 or step == args.steps:  # Prints training loss on the first step, every tenth step, and the final step
            print(f"Step [{step:04d}/{args.steps:04d}] | training loss: {train_loss:.4f}")  # Displays the current step number and training loss

        should_evaluate = (step == 1 or step % args.eval_every == 0 or step == args.steps)  # Decides whether this step should run a train/validation evaluation
        if should_evaluate:  # Runs evaluation only when the evaluation schedule says it is due
            losses = estimate_loss(model, train_data, val_data, args.batch_size,  # Estimates loss on fresh training and validation batches
                                   args.seq_len, mask, args.eval_batches)  # Passes the sequence length, causal mask, and evaluation batch count
            current_train = losses["train"]  # Extracts the estimated training loss from the results dictionary
            current_val = losses["val"]  # Extracts the estimated validation loss from the results dictionary
            print(f"  Evaluation | train loss: {current_train:.4f} | validation loss: {current_val:.4f}")  # Prints the average training and validation losses

        if args.sample_every and (step % args.sample_every == 0 or step == args.steps):  # Checks whether it is time to generate a sample and whether sampling is enabled
            prompt = "First Citizen:"  # Sets the text prompt used to start sample generation
            prompt_tokens = [char_to_int[ch] for ch in prompt if ch in char_to_int]  # Converts prompt characters found in the vocabulary into token IDs
            if not prompt_tokens:  # Checks whether the prompt produced any valid vocabulary tokens
                prompt_tokens = [int(data[0])]  # Uses the dataset's first token as a fallback starting token
            generated_tokens = model.generate(  # Starts autoregressive generation from the prompt token IDs
                prompt_tokens,  # Supplies the starting token sequence to the generator
                max_new_tokens=DEFAULT_GENERATE_TOKENS,  # Sets how many additional characters to generate
                seq_len=args.seq_len,  # Limits the context used for each generation step
                temperature=0.8,  # Uses temperature 0.8 to make sampling somewhat less random
                top_k=min(20, vocab_size),  # Restricts sampling to at most the 20 highest-scoring vocabulary tokens
            )  # Closes the text-generation call
            generated_text = "".join(int_to_char[token] for token in generated_tokens)  # Converts generated token IDs back into characters and joins them into text
            print("\n--- ShaunGPT sample ---")  # Prints a heading before the generated text
            print(generated_text)  # Displays the generated character sequence
            print("-----------------------\n")  # Prints a separator after the generated sample

    print("ShaunGPT Training successfully finished!")  # Displays a message indicating that the training loop has ended

if __name__ == "__main__":  # Runs the following timing and training code only when this file is executed directly
    import time  # Imports Python's time module to measure elapsed runtime
    start_time = time.time()  # Records the start time before training begins
    main()  # Calls the main training function
    elapsed_time = time.time() - start_time  # Calculates total elapsed time since training started
    print(f"\nShaunGPT took {elapsed_time:.2f} seconds to train and test itself.")  # Prints the total runtime in seconds
