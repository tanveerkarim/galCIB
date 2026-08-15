import numpy as np 
from scipy.integrate import simpson 

def ensure_nm_nz_shape(arr, Nm, Nz):
    """
    Ensures arr has shape (Nm, Nz) for broadcasting.

    Parameters
    ----------
    arr : np.ndarray
        Input array with shape (Nm,) or (Nm, Nz) or (Nz,) or scalar.
    Nm : int
        Number of halo mass bins.
    Nz : int
        Number of redshift bins.

    Returns
    -------
    arr : np.ndarray
        Array reshaped to (Nm, Nz).
    """

    if arr.shape == (Nm,):
        arr = arr[:,np.newaxis]  # (Nm, 1)
    return arr

# Power Spectra functions

def compute_Pgg_2h(pkobj):
    """
    Compute P_gg^2h(k,z) = P_lin(k,z)/nbar^2 * [integral(HMF * [Nc+Ns*u]*bias*dlog10Mh)]^2
    
    Also compute magnification bias
    P_gm^2h (k,z) = P_lin(k,z)/nbar * [integral(HMF * [Nc+Ns*u]*bias*dlog10Mh)]
    
    Note P_gg = P_gm * [integral(HMF * [Nc+Ns*u]*bias*dlog10Mh)]/nbar
    """
    
    
    plin = pkobj.cosmo.pk_grid # (Nk, Nz)
    nbar = pkobj.nbar # (Nz,)
    Ig = pkobj.Ig # (Nk,Nz)
    
    Pgmu_2h = plin/nbar[None,:] * Ig
    
    Pgg_2h = plin/pkobj.nbar2 * pkobj.Ig**2
    
    return Pgg_2h, Pgmu_2h

def compute_Pgg_1h(pkobj):
    """
    Returns 1-halo term. 
    
    A10 of 2204.05299.
    
    P_gg_1h = 1/nbar^2 * integral (HMF * [2*Nc*Ns*u + Ns^2*u^2] * dlog10Mh)
    """
    
    prefact = 1/pkobj.nbar2
    numerator = 2*pkobj.ncen*pkobj.nsat_u + (pkobj.nsat_u)**2
    
    integrand = prefact * pkobj.hmf * numerator # (Nk, NMh, Nz)
    integral = simpson(integrand,axis=1,dx=pkobj.dlog10Mh)
    
    Pgg_1h = integral #* prefact 
    
    return Pgg_1h

def compute_Pgmu_2h(pkobj):
    """
    Compute P_gm^2h (k,z) = P_lin(k,z)/nbar * [integral(HMF * [Nc+Ns*u]*bias*dlog10Mh)]
    
    This integral is important if there is magnification bias.
    """
    
    plin = pkobj.cosmo.pk_grid # (Nk, Nz)
    nbar = pkobj.nbar # (Nz,)
    Ig = pkobj.Ig # (Nk,Nz)
    
    p2h = plin/nbar[None,:] * Ig
    
    return p2h

def compute_Pmumu_2h(pkobj):
    """
    Computes the magnification bias auto term.
    This is simply Plin.
    """
    
    return pkobj.cosmo.pk_grid



def compute_PII_2h(pkobj, return_full_matrix=True):
    """
    Compute P_(CIB-CIB)^2h (nu,k,z) = Plin(k,z) * I_CIB(nu) * I_CIB(nu')
        
        I_CIB (nu,k,z) = integral (djc (nu,Mh,z) + djsub(nu,Mh,z)*u(k,Mh,z)) * HMF (Mh,z) * b_halo (Mh,z) * dlog10Mh
    
    Compute using vectorized upper triangle computation.

    Args:
        return_full_matrix: bool
            If True, reconstruct and return the full symmetric (Nnu, Nnu, Nk, Nz) matrix.

    Returns:
        If return_full_matrix:
            P_CIB_full: ndarray of shape (Nnu, Nnu, Nk, Nz)
        Else:
            P_CIB_unique: ndarray of shape (N_unique, Nk, Nz)
            pairs: list of (i, j) tuples corresponding to each unique frequency pair
    """
    
    Nnu, Nk, NMh, Nz = pkobj.djc.shape 
    
    # Vectorized full computation via einsum
    # Result shape: (Nnu, Nnu, Nk, Nz)
    P_II_2h = np.einsum('akz,bkz,kz->abkz', pkobj.Icib, pkobj.Icib, pkobj.cosmo.pk_grid) # (Nnu, Nk, Nz)

    if return_full_matrix:
        return P_II_2h
    else:
        # Extract upper triangle (including diagonal)
        i_idx, j_idx = np.triu_indices(Nnu)
        P_II_2h_unique = P_II_2h[i_idx, j_idx, :, :]
        pairs = list(zip(i_idx, j_idx))
        return P_II_2h_unique, pairs
    
def compute_PII_1h(pkobj, return_full_matrix=False):
    """
    Compute the 1-halo term of the CIB power spectrum using symmetric upper triangle and einsum.
    
    A5 of 2204.05299. 
    
    P_1h (nu,nu') = int(djc*djsub'*u + djc'*djsub*u + djsub*djsub'*u^2)*HMF*dlog10Mh

    Args:
        djc: ndarray (Nnu, Nk, NMh, Nz)
        djsub: ndarray (Nnu, Nk, NMh, Nz)
        u: ndarray (Nk, NMh, Nz)
        hmf: ndarray (NMh, Nz)
        dlog10Mh: scalar
        return_full_matrix: bool

    Returns:
        If return_full_matrix:
            PII_1h_full: ndarray (Nnu, Nnu, Nk, Nz)
        Else:
            PII_1h_unique: ndarray (N_unique, Nk, Nz)
            pairs: list of (i, j)
    """
    Nnu, Nk, NMh, Nz = pkobj.djc_plus_djsub_u.shape
    
    i_idx, j_idx = np.triu_indices(Nnu)
    pairs = list(zip(i_idx, j_idx))

    PII_1h_unique = np.empty((len(pairs), Nk, Nz))

    for idx, (i, j) in enumerate(pairs):
        term1 = pkobj.djc[i] * pkobj.djsub_u[j] # (Nk, NMh, Nz)
        term2 = pkobj.djc[j] * pkobj.djsub_u[i]
        term3 = pkobj.djsub_u[i] * pkobj.djsub_u[j]

        integrand = (term1 + term2 + term3) * pkobj.hmf[None,:,:]  # shape (Nk, NMh, Nz)
        PII_1h_unique[idx] = simpson(integrand, 
                                     dx=pkobj.dlog10Mh, 
                                     axis=1)  # shape (Nk, Nz)

    if return_full_matrix:
        PII_1h_full = np.zeros((Nnu, Nnu, Nk, Nz))
        for idx, (i, j) in enumerate(pairs):
            PII_1h_full[i, j] = PII_1h_unique[idx]
            if i != j:
                PII_1h_full[j, i] = PII_1h_unique[idx]
        return PII_1h_full
    else:
        return PII_1h_unique, pairs
    
def compute_PgI_2h(pkobj):
    """
    Returns 2-halo term of galaxy X CIB. 
    
    A14 of 2204.05299.
    
    P_gI(nu,k,z) = Plin/nbar * integral (HMF*[nc+ns*u]*bias*dlog10Mh) * integral(HMF*(djc+djsub*u)*bias*dlog10Mh)
    Note, P_gI = P_muI/nbar * integral (HMF*[nc+ns*u]*bias*dlog10Mh) 
    
    Added magnification bias correction too. 
    P_muI(nu,k,z) = Plin * integral(HMF*(djc+djsub*u)*bias*dlog10Mh)
    """
    
    PmuI_2h = pkobj.Icib * pkobj.cosmo.pk_grid
    PgI_2h = pkobj.Ig * PmuI_2h/pkobj.nbar 
    
    return PgI_2h, PmuI_2h

def compute_PgI_1h(pkobj):
    """
    Returns 1-halo term of galaxy X CIB.
    
    A13 of 2204.05299.
    
    P(nu,k,z) = 1/nbar * integral (HMF * (nc+ns*u) * (djc+djsub*u) * dlog10Mh)
    """

    galterm = pkobj.ncen_plus_nsat_u
    cibterm = pkobj.djc_plus_djsub_u
    
    integrand = pkobj.hmf * galterm * cibterm # (Nnu,Nk,NMh,Nz)
    
    Pk_1h = 1/pkobj.nbar * simpson(integrand, dx=pkobj.dlog10Mh, axis=2)
    
    return Pk_1h

def compute_Puv_tot(pk1h, pk2h, hmalpha=1):
    """
    Returns the total P(k) of the fields U and V.
    
    Optional hmalpha decides the softening of the halo transition scale:
    
    P_tot = (pk2h^alpha + pk1h^alpha)^(1/alpha)
    """
    # NOTE: the parentheses matter -- `x**1/hmalpha` parses as `(x**1)/hmalpha`
    # because ** binds tighter than /. That happened to be correct only at
    # hmalpha == 1; analysis/theory.py passes 0.7.
    P_tot = (pk1h**hmalpha + pk2h**hmalpha) ** (1.0 / hmalpha)

    return P_tot
