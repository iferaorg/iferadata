"""
Masked tensor operations for financial time series data.
Provides utility functions for handling missing data in time series using regular tensors with separate masks.
"""

from typing import Optional, Tuple

import torch
from einops import rearrange, repeat

from .series import ema, rtr, sma


def compress_tensor(t, mask):
    """
    Compresses tensor t along its last dimension according to the boolean mask.
    Valid elements (where mask is True) are moved to the beginning (preserving order)
    and the remaining positions are filled with NaNs.
    Args:
        t (torch.Tensor): Input tensor of shape [..., n].
        mask (torch.BoolTensor): Boolean mask of the same shape as t.

    Returns:
        torch.Tensor: Compressed tensor of the same shape as t.
    """
    mask = mask.bool()
    # Compute cumulative count indices along the last dimension.
    # For each valid element, its new position is given by its (cumulative count - 1).
    valid_indices = mask.cumsum(dim=-1) - 1
    # Remember the original batch shape and last-dim size.
    original_batch_shape = t.shape[:-1]
    n = t.shape[-1]
    # Flatten all batch dimensions into one dimension using einops.
    flat_t = rearrange(t, "... n -> (...) n")
    flat_mask = rearrange(mask, "... n -> (...) n")
    flat_valid_indices = rearrange(valid_indices, "... n -> (...) n")
    # Number of flattened rows.
    b, _ = flat_t.shape
    # Create an output tensor filled with NaNs.
    flat_compressed = torch.full_like(flat_t, float("nan"))
    # Create row indices for all entries.
    row_idx = repeat(torch.arange(b, device=t.device), "b -> b n", n=n)
    # Scatter valid elements into their new positions.
    flat_compressed[row_idx[flat_mask], flat_valid_indices[flat_mask]] = flat_t[
        flat_mask
    ]
    # Restore the original shape.
    return flat_compressed.view(*original_batch_shape, n)


def decompress_tensor(compressed, mask):
    """
    Decompresses a tensor that was compressed with compress_tensor,
    restoring valid elements to their original positions.
    Args:
        compressed (torch.Tensor): Compressed tensor of shape [..., n].
        mask (torch.BoolTensor): The original boolean mask of shape [..., n].

    Returns:
        torch.Tensor: Decompressed tensor of the same shape, with valid entries restored and
                      other positions as NaN.
    """
    mask = mask.bool()
    valid_indices = mask.cumsum(dim=-1) - 1
    original_batch_shape = compressed.shape[:-1]
    n = compressed.shape[-1]
    flat_compressed = rearrange(compressed, "... n -> (...) n")
    flat_mask = rearrange(mask, "... n -> (...) n")
    flat_valid_indices = rearrange(valid_indices, "... n -> (...) n")
    b, _ = flat_compressed.shape
    flat_decompressed = torch.full_like(flat_compressed, float("nan"))
    row_idx = repeat(torch.arange(b, device=compressed.device), "b -> b n", n=n)
    # For each valid position in the original tensor, recover the element
    # from the compressed tensor.
    flat_decompressed[flat_mask] = flat_compressed[
        row_idx[flat_mask], flat_valid_indices[flat_mask]
    ]
    return flat_decompressed.view(*original_batch_shape, n)


def masked_sma(data: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
    """
    Calculates a simple moving average (SMA) on masked data, respecting the mask.
    The function compresses the tensor according to the mask, performs the SMA calculation,
    and then decompresses the result back to the original shape.
    Args:
        data (torch.Tensor): Input data tensor of shape [..., n].
        mask (torch.Tensor): Boolean mask of shape [..., n] (True for valid values).
        window (int): Window size for the simple moving average calculation.

    Returns:
        - torch.Tensor: Tensor containing the SMA values with same shape as input.
    """
    # Compress the tensor along the last dimension according to the mask.
    cdata = compress_tensor(data, mask).nan_to_num(nan=0.0)

    # Calculate the SMA of the compressed tensor.
    cresult = sma(cdata, window)
    # Decompress the result tensor to restore the original shape.
    rdata = decompress_tensor(cresult, mask)

    return rdata


def masked_ema(
    data: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Calculates an exponential moving average (EMA) on masked data, respecting the mask.
    The function compresses the tensor according to the mask, performs the EMA calculation,
    and then decompresses the result back to the original shape.
    Args:
        data (torch.Tensor): Input data tensor of shape [..., n].
        mask (torch.Tensor): Boolean mask of shape [..., n] (True for valid values).
        alpha (float): Smoothing factor between 0 and 1.
        chunk_size (Optional[int], default=None): If provided, process the series
                                                  in chunks of this size to reduce memory usage.

    Returns:
        - torch.Tensor: Tensor containing the EMA values with same shape as input.
    """
    # Compress the tensor along the last dimension according to the mask.
    cdata = compress_tensor(data, mask).nan_to_num(nan=0.0)

    # Calculate the EMA of the compressed tensor.
    cresult = ema(cdata, alpha, chunk_size)
    # Decompress the result tensor to restore the original shape.
    rdata = decompress_tensor(cresult, mask)

    return rdata


def masked_rtr(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Calculates the relative true range (RTR) on masked data, respecting the mask.
    The function compresses the tensor according to the mask, performs the RTR calculation,
    and then decompresses the result back to the original shape.

    Args:
        data (torch.Tensor): Input data tensor of shape [..., n, channels] where channels
                          contains [open, high, low, close, volume].
        mask (torch.Tensor): Boolean mask of shape [..., n] (True for valid values).

    Returns:
        - torch.Tensor: Tensor containing the RTR values with the same shape
                        (except for the removed channels dimension).
    """
    # Make time the last dimension for compress_tensor
    data_cn = rearrange(data, "... n c -> ... c n")
    mask_cn = repeat(mask, "... n -> ... c n", c=data_cn.shape[-2])

    # Compress the tensor along the time dimension according to the mask
    cdata = compress_tensor(data_cn, mask_cn).nan_to_num(
        nan=1.0
    )  # Avoid division by zero
    cdata = rearrange(cdata, "... c n -> ... n c")

    # Calculate the RTR of the compressed tensor
    cresult = rtr(cdata)

    # Decompress the result tensor to restore the original shape
    rdata = decompress_tensor(cresult, mask)

    return rdata


def masked_artr(
    data: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    acrossday: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Calculates the average relative true range (ARTR) on masked data, respecting the mask.
    The function compresses the tensor according to the mask, performs the ARTR calculation,
    and then decompresses the result back to the original shape.

    Args:
        data (torch.Tensor): Input data tensor of shape [..., date, time, channels].
        mask (torch.Tensor): Boolean mask of shape [..., date, time] (True for valid values).
        alpha (float): Smoothing factor for the EMA.
        acrossday (bool, optional): If True, calculate ARTR across days. Defaults to False.
        chunk_size (Optional[int], optional): If provided, calculate the EMA in chunks of this size.
                                             Defaults to None.

    Returns:
        - torch.Tensor: Tensor containing the ARTR values with same shape as RTR result.
    """
    rtr_data = masked_rtr(data, mask)

    if acrossday:
        # Flatten the date and time dimensions into one
        original_shape = rtr_data.shape
        rtr_flat = rtr_data.view(*rtr_data.shape[:-2], -1)
        rtr_mask_flat = mask.view(*mask.shape[:-2], -1)

        # Calculate the ARTR
        artr_flat = masked_ema(rtr_flat, rtr_mask_flat, alpha, chunk_size)

        # Reshape back to original dimensions
        artr_data = artr_flat.view(*original_shape)

        return artr_data

    # Calculate the ARTR for each date separately
    return masked_ema(rtr_data, mask, alpha, chunk_size)
