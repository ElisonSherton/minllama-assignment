from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def implement_rope_equation(
    real: torch.Tensor, imag: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    odd_position_elements = real * cosine - imag * sine
    even_position_elements = imag * cosine + real * sine

    even_position_elements = even_position_elements.unsqueeze(-1)
    odd_position_elements = odd_position_elements.unsqueeze(-1)

    stacked_tensor = torch.stack(
        (odd_position_elements, even_position_elements), dim=-1
    )
    final_tensor = stacked_tensor.reshape(*stacked_tensor.shape[:3], -1)
    return final_tensor


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = (
        query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    )
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Create theta vector and sequence vector
    dby2 = head_dim // 2
    theta_vector = torch.pow(
        torch.ones(dby2, dtype=query.dtype, device=device) * theta,
        -2 * (torch.arange(1, dby2 + 1, device=device) - 1) / head_dim,
    )
    sequence_vector = torch.arange(seqlen, dtype=query.dtype, device=device)
    m_theta_vector = sequence_vector.unsqueeze(1) @ theta_vector.unsqueeze(0)
    # Right now, m_theta_vector shape is sequence_len * head_dim // 2
    # We want it to be batch_size, seq_len, num_heads, head_dim
    # Hence unsqueeze along the zeroth (batch) and then along the second dimension (head)
    m_theta_vector = m_theta_vector.unsqueeze(0).unsqueeze(2)
    cos_theta = torch.cos(m_theta_vector)
    sin_theta = torch.sin(m_theta_vector)

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_out = implement_rope_equation(query_real, query_imag, cos_theta, sin_theta)
    key_out = implement_rope_equation(key_real, key_imag, cos_theta, sin_theta)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out