def evolving_log_mass(mu0, mup, z_over_1plusz):
    """
    Parametrize redshift evolution of log10 mass:
    mu(z) = mu0 + (z / (1 + z)) * mup

    Args
    ----------
    mu_0 : float
        Base log10(M) at z=0.
    mu_p : float
        Evolution parameter.
    z_over_1plusz : array_like or float
        z / (1+z), precomputed for efficiency.

    Returns
    -------
    float or array_like
        The evolving log mass at each redshift.
    """

    return mu0 + z_over_1plusz * mup
