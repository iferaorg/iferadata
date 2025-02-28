import torch
from einops import rearrange
from typing import Optional


def sma(t: torch.Tensor, window: int) -> torch.Tensor:
    """
    Calculate the simple moving average of a tensor. For the first window-1 elements,
    the average is over all elements up to that position. For subsequent elements,
    the average is over the last window elements.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor. The last dimension is the one to calculate the moving average over.
    window : int
        Window size.

    Returns
    -------
    y : torch.Tensor
        Output tensor with the same shape as x.
    """
    n = min(t.shape[-1], window)

    # Compute sma for the elements from the n-th to the last
    tail = t.unfold(dimension=-1, size=n, step=1).mean(dim=-1)

    # Compute the average for the first n-1 elements    
    head_nom = torch.cumsum(t[..., :n-1], dim=-1)
    head_denom = torch.arange(1, n, device=t.device, dtype=t.dtype)
    head = head_nom / head_denom
    
    return torch.cat([head, tail], dim=-1)


def ema(x: torch.Tensor, alpha: float, chunk_size: Optional[int] = None) -> torch.Tensor:
    """
    Calculate the exponential moving average of a tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. The last dimension is the one to calculate the moving average over.
    alpha : float
        Smoothing factor between 0 and 1.
    chunk_size : Optional[int], default=None
        If provided, process the series in chunks of this size to reduce memory usage.

    Returns
    -------
    y : torch.Tensor
        Output tensor with the same shape as x, containing the EMA along the last dimension.
    """
    # Get tensor properties
    shape = x.shape
    n = shape[-1]
    device = x.device
    dtype = x.dtype

    if chunk_size is None or chunk_size >= n:
        # Use original method for small series or if chunk_size not specified
        i = torch.arange(n, device=device).unsqueeze(0)  # Shape: (1, n)
        j = torch.arange(n, device=device).unsqueeze(1)  # Shape: (n, 1)
        diff = i - j  # Shape: (n, n)
        W = torch.where(diff >= 0, (1 - alpha) ** diff, torch.zeros_like(diff, dtype=dtype))
        W[1:, :] *= alpha  # First row unchanged (W[0, 0] = 1), others scaled
        return x @ W
    else:
        # Process in chunks to save memory
        chunks = torch.split(x, chunk_size, dim=-1)  # Split along last dimension
        y_chunks = []
        y_prev = torch.Tensor()

        for idx, x_chunk in enumerate(chunks):
            m = x_chunk.shape[-1]  # Size of current chunk
            i = torch.arange(m, device=device).unsqueeze(0)
            j = torch.arange(m, device=device).unsqueeze(1)
            diff = i - j
            W_local = torch.where(diff >= 0, (1 - alpha) ** diff, 
                                 torch.zeros_like(diff, dtype=dtype))

            if idx == 0:
                # First chunk: mimic original EMA starting condition
                W_local[1:, :] *= alpha  # W_local[0, 0] = 1, rest scaled
            else:
                # Subsequent chunks: compute local EMA, all rows scaled
                W_local *= alpha

            y_local = x_chunk @ W_local

            if idx > 0:
                # Add decayed contribution from previous chunk
                decay = (1 - alpha) ** torch.arange(1, m + 1).to(device)  # Shape: (m,)
                y_chunk = y_local + decay * y_prev[..., None]  # Broadcasting: (..., m)
            else:
                y_chunk = y_local

            y_chunks.append(y_chunk)
            y_prev = y_chunk[..., -1]  # Shape: (...), last EMA value for next chunk

        return torch.cat(y_chunks, dim=-1)  # Concatenate along last dimension


def ema_slow(x: torch.Tensor, alpha: float):
    """
    Calculate the exponential moving average of a tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. The last dimension is the one to calculate the moving average over.
    alpha : float
        Smoothing factor between 0 and 1.
    
    Returns
    -------
    y : torch.Tensor
        Output tensor with the same shape as x, containing the EMA along the last dimension.
    """
    y = torch.zeros_like(x)
    y[..., 0] = x[..., 0]
    for i in range(1, x.size(-1)):
        y[..., i] = (1.0 - alpha) * y[..., i - 1] + alpha * x[..., i]
    return y
        

def ffill(t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Forward fill missing values in a tensor along the last dimension.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor.
    mask : Optional[torch.Tensor]
        Boolean mask tensor indicating valid (non-missing) values. If None, defaults to ~torch.isnan(t).

    Returns
    -------
    t_ffilled : torch.Tensor
        Output tensor with missing values forward filled.
    """
    indices = torch.arange(t.size(-1), device=t.device, dtype=torch.int64).expand_as(t)
    mask = mask if mask is not None else ~torch.isnan(t)
    valid_indices = indices.masked_fill(~mask, 0)
    cummax_indices, _ = valid_indices.cummax(dim=-1)
    
    return torch.gather(t, dim=-1, index=cummax_indices)


def rtr(t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the relative true range of a tensor along the second to last dimension.
    Definition: RTR = MAX(high, prev_close) / MIN(low, prev_close) - 1; high / low - 1 for the first element.
    
    Parameters
    ----------
    t : torch.Tensor
        Input tensor.
        Last dimension is channels: [open, high, low, close, volume].
        Second to last dimension is the time dimension. Shape: (..., time, channels).
    
    Returns
    -------
    rtr_t : torch.Tensor
        Output tensor with shape (..., time). NaN values in input will result in NaN values in output.
    """
    # Extract channels
    high = t[..., :, 1]   # Shape: (..., time)
    low = t[..., :, 2]    # Shape: (..., time)
    close = t[..., :, 3]  # Shape: (..., time)

    # Create previous close by shifting close right, with NaN at t=0
    prev_close = close[..., :-1]  # Shape: (..., time)

    # First element: high / low - 1
    # Other elements: MAX(high, prev_close) / MIN(low, prev_close) - 1
    max_part = torch.max(high[..., 1:], prev_close)  # Shape: (..., time)
    min_part = torch.min(low[..., 1:], prev_close)   # Shape: (..., time)
    
    # Calculate RTR - NaN values will propagate naturally
    rtr_general = max_part / min_part - 1  # Shape: (..., time)
    
    return torch.cat([high[..., 0:1] / low[..., 0:1] - 1, rtr_general], dim=-1)


def artr(t: torch.Tensor, alpha: float, acrossday: bool = False, chunk_size: Optional[int] = None) -> torch.Tensor:
    """
    Calculate the average relative true range of a tensor along the second to last dimension.
    Definition: ARTR = EMA(RTR, alpha).
    
    Parameters
    ----------
    t : torch.Tensor
        Input tensor. Shape: (..., date, time, channels).
        Last dimension is channels: [open, high, low, close, volume].
    alpha : float
        Smoothing factor for the EMA.
    acrossday : bool, default=False
        If True, calculate ARTR across days, i.e. calculate on a continuous date+time series without resetting at the start of each day.
        If False, calculate ARTR for each date separately.
    chunk_size : Optional[int], default=None
        If provided, calculate the EMA in chunks of this size to reduce memory usage.

    Returns
    -------
    artr_t : torch.Tensor
        Output tensor with shape (..., date, time). Contains NaN for invalid
        
    """
    # Calculate RTR
    rtr_t = rtr(t)  # Shape: (..., date, time)
    
    if acrossday:
        # Flatten date and time dimensions
        rtr_flat = rearrange(rtr_t, '... d t -> ... (d t)')
        
        # Calculate ARTR
        artr_flat = ema(rtr_flat, alpha, chunk_size=chunk_size)
        
        # Reshape to original shape
        return rearrange(artr_flat, '... (d t) -> ... d t', t=rtr_t.size(-1))
    else:
        # Calculate ARTR for each date
        artr_t = ema(rtr_t, alpha, chunk_size)  # Shape: (..., date, time)
        return artr_t
    