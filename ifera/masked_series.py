"""
Masked tensor operations for financial time series data.
Provides utility functions for handling missing data in time series using PyTorch's MaskedTensor.
"""

import torch
from .series import sma, ema, rtr
from torch.masked import masked_tensor, MaskedTensor
from einops import rearrange, repeat
from typing import Optional, cast


def ohlcv_to_masked(ohlcv_data: torch.Tensor) -> MaskedTensor:
    """
    Converts OHLCV (Open, High, Low, Close, Volume) data to a MaskedTensor.
    The function creates a mask based on the volume data, where zero volume
    indicates missing data.

    Args:
        ohlcv_data (torch.Tensor): Input OHLCV tensor of shape [..., date, time, 5].

    Returns:
        MaskedTensor: Masked tensor containing the OHLCV data with the same shape.
    """
    volume = ohlcv_data[..., -1]
    mask = volume.to(torch.int) != 0
    mask = repeat(mask, "... -> ... c", c=ohlcv_data.shape[-1])
    return masked_tensor(ohlcv_data, mask)


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


def masked_sma(t: MaskedTensor, window: int) -> MaskedTensor:
    """
    Calculates a simple moving average (SMA) on a MaskedTensor, respecting the mask.
    The function compresses the tensor according to the mask, performs the SMA calculation,
    and then decompresses the result back to the original shape.
    Args:
        t (MaskedTensor): Input masked tensor of shape [..., n].
        window (int): Window size for the simple moving average calculation.

    Returns:
        MaskedTensor: Masked tensor containing the SMA values with the same 
        shape and mask as the input.
    """
    data = t.get_data()
    mask = t.get_mask()

    # Compress the tensor along the last dimension according to the mask.
    cdata = compress_tensor(data, mask).nan_to_num(nan=0.0)

    # Calculate the SMA of the compressed tensor.
    cresult = sma(cdata, window)
    # Decompress the result tensor to restore the original shape.
    rdata = decompress_tensor(cresult, mask)

    return masked_tensor(rdata, mask)


def masked_ema(
    t: MaskedTensor, alpha: float, chunk_size: Optional[int] = None
) -> MaskedTensor:
    """
    Calculates an exponential moving average (EMA) on a MaskedTensor, respecting the mask.
    The function compresses the tensor according to the mask, performs the EMA calculation,
    and then decompresses the result back to the original shape.
    Args:
        t (MaskedTensor): Input masked tensor of shape [..., n].
        alpha (float): Smoothing factor between 0 and 1.
        chunk_size (Optional[int], default=None): If provided, process the series
                                                  in chunks of this size to reduce memory usage.

    Returns:
        MaskedTensor: Masked tensor containing the EMA values with the 
        same shape and mask as the input.
    """
    data = t.get_data()
    mask = t.get_mask()

    # Compress the tensor along the last dimension according to the mask.
    cdata = compress_tensor(data, mask).nan_to_num(nan=0.0)

    # Calculate the EMA of the compressed tensor.
    cresult = ema(cdata, alpha, chunk_size)
    # Decompress the result tensor to restore the original shape.
    rdata = decompress_tensor(cresult, mask)

    return masked_tensor(rdata, mask)


def masked_rtr(t: MaskedTensor) -> MaskedTensor:
    """
    Calculates the relative true range (RTR) on a MaskedTensor, respecting the mask.
    The function compresses the tensor according to the mask, performs the RTR calculation,
    and then decompresses the result back to the original shape.

    Args:
        t (MaskedTensor): Input masked tensor of shape [..., n, channels] where channels
                          contains [open, high, low, close, volume].

    Returns:
        MaskedTensor: Masked tensor containing the RTR values with the same shape (except for
                      the removed channels dimension) and mask as the input.
    """
    data = t.get_data()
    mask = t.get_mask()

    # Make time the last dimension for compress_tensor
    data_cn = rearrange(data, "... n c -> ... c n")
    mask_cn = rearrange(mask, "... n c -> ... c n")

    # Compress the tensor along the time dimension according to the mask
    cdata = compress_tensor(data_cn, mask_cn).nan_to_num(
        nan=1.0
    )  # Avoid division by zero
    cdata = rearrange(cdata, "... c n -> ... n c")

    # Calculate the RTR of the compressed tensor
    cresult = rtr(cdata)

    # The result mask should match the shape of cresult, which has one less dimension
    # than the input (channels dimension is removed)
    result_mask = mask.any(dim=-1)

    # Decompress the result tensor to restore the original shape
    rdata = decompress_tensor(cresult, result_mask)

    return masked_tensor(rdata, result_mask)


def masked_artr(
    t: MaskedTensor,
    alpha: float,
    acrossday: bool = False,
    chunk_size: Optional[int] = None,
) -> MaskedTensor:
    """
    Calculates the average relative true range (ARTR) on a MaskedTensor, respecting the mask.
    The function compresses the tensor according to the mask, performs the ARTR calculation,
    and then decompresses the result back to the original shape.

    Args:
        t (MaskedTensor): Input masked tensor of shape [..., date, time, channels].
        alpha (float): Smoothing factor for the EMA.
        acrossday (bool, optional): If True, calculate ARTR across days. Defaults to False.
        chunk_size (Optional[int], optional): If provided, calculate the EMA in chunks of this size.
                                              Defaults to None.

    Returns:
        MaskedTensor: Masked tensor containing the ARTR values with the same shape (except for
                      the removed channels dimension) and mask as the input.
    """
    rtr_t = masked_rtr(t)
    if acrossday:
        # Flatten the date and time dimensions into one. Use cast for Pylance.
        rtr_flat = cast(MaskedTensor, rtr_t.view(*rtr_t.shape[:-2], -1))

        # Calculate the ARTR
        artr_flat = masked_ema(rtr_flat, alpha, chunk_size)

        return cast(MaskedTensor, artr_flat.view_as(rtr_t))

    # Calculate the ARTR for each date separately
    return masked_ema(rtr_t, alpha, chunk_size)
