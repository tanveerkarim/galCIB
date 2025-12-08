import os
import re
import numpy as np

def load_my_filters(filter_dir, nu_obs=None):
    """
    Load a subset of Planck filters from directory using regex pattern matching.

    Parameters
    ----------
    filter_dir : str
        Directory containing the filter response files.
    nu_obs : list of float, optional
        Frequencies in GHz to load. If None, loads all available.

    Returns
    -------
    filters : dict
        Dictionary mapping frequency (GHz) to (freq_array_Hz, response_array)
    """
    filters = {}

    # Regex to capture the number after "HFI__avg_" (e.g. 857)
    freq_pattern = re.compile(r"HFI__avg_(\d+)_")

    for fname in os.listdir(filter_dir):
        match = freq_pattern.search(fname)
        if not match:
            continue

        freq = float(match.group(1))  # frequency in GHz

        if nu_obs is not None and freq not in nu_obs:
            continue

        filepath = os.path.join(filter_dir, fname)
        data = np.loadtxt(filepath, comments="#")

        freq_ghz = data[:,1]  # freq column in GHz
        response = data[:,2]  # normalized response column
        freq_hz = freq_ghz * 1e9  # convert GHz to Hz

        filters[freq] = (freq_hz, response)

    return filters
