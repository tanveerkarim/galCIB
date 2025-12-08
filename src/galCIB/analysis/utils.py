import numpy as np


def bin_mat(r, mat, r_bins, axis=-1):
    """
    Bin `mat` along the specified `axis` using bin edges defined by `r_bins`.

    Parameters
    ----------
    r : array_like
        1D array corresponding to the axis being binned (e.g., ell or radius).
    mat : ndarray
        Input array with shape (..., len(r), ...), to be binned along `axis`.
    r_bins : array_like
        Bin edges for the `r` array.
    axis : int, optional
        Axis of `mat` to bin over (default: -1, last axis).

    Returns
    -------
    bin_center : ndarray
        Midpoints of the bins.
    mat_binned : ndarray
        Binned array with shape where the length along `axis` is len(r_bins)-1.
    """
    r = np.asarray(r)
    r_bins = np.asarray(r_bins)

    bin_idx = np.digitize(r, r_bins) - 1
    n_bins = len(r_bins) - 1
    bin_center = 0.5 * (r_bins[1:] + r_bins[:-1])

    # Compute weights (r * dr)
    r2 = np.sort(np.unique(np.append(r, r_bins)))
    dr = np.gradient(r2)
    r2_idx = np.searchsorted(r2, r)
    dr = dr[r2_idx]
    weights = r * dr  # shape: (Nr,)

    # Move axis to last for easier indexing
    mat = np.moveaxis(mat, axis, -1)  # shape becomes (..., Nr)
    orig_shape = mat.shape[:-1]
    # Nr = mat.shape[-1]

    # Apply weights
    mat_w = mat * weights  # broadcasted multiplication

    mat_binned = np.zeros(orig_shape + (n_bins,), dtype=float)
    norm_binned = np.zeros(orig_shape + (n_bins,), dtype=float)

    for i in range(n_bins):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        mat_binned[..., i] = np.sum(mat_w[..., mask], axis=-1)
        norm_binned[..., i] = np.sum(weights[mask])

    # Normalize
    with np.errstate(divide="ignore", invalid="ignore"):
        mat_binned = np.divide(mat_binned, norm_binned, where=norm_binned > 0)

    # Move binned axis back to original location
    mat_binned = np.moveaxis(mat_binned, -1, axis)

    return bin_center, mat_binned
