import numpy as np
import torch

# Embed table: (seq_len, d_model)
# For each position pos and each dimension i,
#    PE(pos, 2i) = sin(pos / 10000^(2i / d_model))
#    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

def get_sinusoidal_positional_embeddings(seq_len, d_model):
    """
    Generate sinusoidal positional embeddings.

    Args:
        seq_len (int): Length of the sequence.
        d_model (int): Dimension of the embeddings.

    Returns:
        torch.Tensor: Positional embeddings of shape (seq_len, d_model).
    """
    # Initialize the positional embedding matrix
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Compute the positional embeddings
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    # Convert to a PyTorch tensor
    pe = torch.tensor(pe, dtype=torch.float32)
    
    return pe