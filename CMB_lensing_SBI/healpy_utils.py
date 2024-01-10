import healpy as hp
import numpy as np

def gk_inv(K, KB, nside, lmax):
    """
    Perform an inverse transformation from convergence maps K and KB to shear maps.

    Args:
    K (array-like): Input convergence map.
    KB (array-like): Input B-mode convergence map.
    nside (int): HEALPix resolution parameter.
    lmax (int): Maximum multipole number.

    Returns:
    tuple: Contains two arrays representing the transformed E and B mode shear maps.
    """

    # Convert the convergence map K to spherical harmonics coefficients
    alms = hp.map2alm(K, lmax=lmax, pol=False)

    # Retrieve the multipole indices for spherical harmonics
    ell, emm = hp.Alm.getlm(lmax=lmax)

    # Perform a specific transformation for E-mode
    kalmsE = alms / (1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
    kalmsE[ell == 0] = 0.0  # Setting the transformed coefficients to zero for ell=0

    # Repeat the process for the KB map
    alms = hp.map2alm(KB, lmax=lmax, pol=False)
    ell, emm = hp.Alm.getlm(lmax=lmax)
    kalmsB = alms / (1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
    kalmsB[ell == 0] = 0.0  # Setting the transformed coefficients to zero for ell=0

    # Convert the transformed spherical harmonics coefficients back to a map
    _, e1t, e2t = hp.alm2map([kalmsE, kalmsE, kalmsB], nside=nside, lmax=lmax, pol=True)
    
    return e1t, e2t


def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048, nosh=True):
    """
    Convert shear to convergence on a sphere. Inputs are all healpix maps.

    Args:
    gamma1 (array-like): Input shear map (component 1).
    gamma2 (array-like): Input shear map (component 2).
    mask (array-like): Mask to apply on the shear maps.
    nside (int, optional): HEALPix resolution parameter. Defaults to 1024.
    lmax (int, optional): Maximum multipole number. Defaults to 2048.
    nosh (bool, optional): Flag to apply a specific transformation. Defaults to True.

    Returns:
    tuple: Contains E and B mode convergence maps, and the transformed spherical harmonics coefficients.
    """

    # Apply the mask to the shear maps
    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    # Combine masked maps
    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]

    # Convert the masked maps to spherical harmonics coefficients
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)

    # Retrieve the multipole indices for spherical harmonics
    ell, emm = hp.Alm.getlm(lmax=lmax)

    # Apply transformations to the spherical harmonics coefficients based on the 'nosh' flag
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1]
        almsB = alms[2]
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0

    # Convert the transformed spherical harmonics coefficients back to maps
    kappa_map_alm = hp.alm2map(alms[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almsE, nside=nside, lmax=lmax, pol=False)
    
    
def IndexToDeclRa(index, nside, nest=False):
    """
    Convert HEALPix index to Declination and Right Ascension.

    Args:
    index (int): HEALPix index.
    nside (int): HEALPix resolution parameter.
    nest (bool, optional): Nesting flag for HEALPix. Defaults to False.

    Returns:
    tuple: Declination and Right Ascension corresponding to the given HEALPix index.
    """

    # Convert index to angular coordinates
    theta, phi = hp.pixelfunc.pix2ang(nside, index, nest=nest)

    # Convert angular coordinates to Declination and Right Ascension
    return -np.degrees(theta - np.pi / 2.), np.degrees(phi)


def convert_to_pix_coord(ra, dec, nside=1024):
    """
    Converts Right Ascension and Declination to HEALPix coordinates.

    Args:
    ra (float): Right Ascension.
    dec (float): Declination.
    nside (int, optional): HEALPix resolution parameter. Defaults to 1024.

    Returns:
    int: HEALPix pixel index corresponding to the given Right Ascension and Declination.
    """

    # Convert RA and Dec to theta and phi
    theta = (90.0 - dec) * np.pi / 180.
    phi = ra * np.pi / 180.

    # Convert angular coordinates to HEALPix index
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    return pix