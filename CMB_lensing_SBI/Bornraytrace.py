import healpy as hp
from scipy.integrate import trapz
from scipy.integrate import simps
from astropy import constants as const
import numpy as np
from astropy import units as u


def kappa_prefactor(H0, om0, length_unit='Mpc'):
    """
    Gives prefactor (3 H_0^2 Om0)/2

    :param H0: Hubble parameter with astropy units
    :param om0: Omega matter
    :param length_unit: for H0 (default Mpc)
    :return: prefactor for lensing

    """

    bit_with_units = H0.to(u.s ** -1)/const.c.to(str(length_unit + '/s'))

    return 1.5 * om0 * bit_with_units * bit_with_units


def raytrace_integration(kappa_prefactor, overdensity_array, a_centre, comoving_edges, mask=None, old_approach = False):
    """
    This function evaluates the Born weak lensing integral

    :param kappa_prefactor: defined as the output of the function kappa_prefactor
    :param overdensity_array: an 2D array of overdensity healpix maps in radial shells
    :param a_centre: scale factor at comoving centre of shells
    :param comoving_edges: comoving distance to edges of shells
    :param mask: healpix map where 1 is observed and 0 is mask
    :return: convergence kappa map
    """

    assert overdensity_array.shape[1] + 1 == comoving_edges.shape[0]

    dr_array = comoving_edges[1:] - comoving_edges[:-1]

    comoving_centre = 0.5*(comoving_edges[:-1] + comoving_edges[1:])
    if old_approach:
        # this was the Born implementation by Niall.
        comoving_max = comoving_edges[-1]
    else:
        comoving_max = comoving_centre[-1]
    
    comoving_prefactors = dr_array * (comoving_max - comoving_centre) * comoving_centre / (comoving_max * a_centre)
    comoving_prefactors *= kappa_prefactor

    if mask is not None:
        mask = np.where(mask>0.5,1.,0.).T
        overdensity_array = (mask * overdensity_array.T).T

    return np.sum(comoving_prefactors * overdensity_array,axis=1).value

  
def raytrace(H0, om0, overdensity_array, a_centre, comoving_edges, mask=None, Hubble_length_unit = 'Mpc', old_approach = False):
    """
    Evaluate weak lensing convergence map using Born approximation

    :param H0: Hubble parameter with astropy units
    :param om0: Omega matter
    :param overdensity_array: an 2D array of overdensity healpix maps in radial shells
    :param a_centre: scale factor at comoving centre of shells
    :param comoving_edges: comoving distance to edges of shells
    :param mask: healpix map where 1 is observed and 0 is mask
    :param length_unit: for H0 (default Mpc)
    :return: convergence kappa map
    """

    kappa_pref_evaluated = kappa_prefactor(H0, om0, length_unit = Hubble_length_unit)

    kappa_raytraced = raytrace_integration(kappa_pref_evaluated, overdensity_array, a_centre, comoving_edges, mask, old_approach = old_approach)

    return kappa_raytraced


def W_kernel(r_array, z_array, nz, simpsons=False):
    """
    lensing kernel W s.t.  kappa = prefactor * integral  W(r) * overdensity(r)  dr

    :param r_array: comoving distances array
    :param z_array: redshift array matching r_array (cosmology dependent)
    :param nz: source redshift distribution
    :param simpsons: boolean to use simpsons integratio
    :return: W = r * q /r
    """

    # normalised redshift distribution nr
    if simpsons:
        normalisation = simps(nz, r_array)
    else:
        normalisation = trapz(nz, r_array)

    nr = nz / normalisation

    q = np.empty(r_array.shape)  # q efficiency  eq. 24 in Kilbinger 15
    for i in range(len(r_array)):
        r = r_array[i]
        integrand = np.multiply(np.divide(r_array[i:] - r, r_array[i:]), nr[i:])
        if simpsons:
            q[i] = simps(integrand, r_array[i:])
        else:
            q[i] = trapz(integrand, r_array[i:])

    return q * r_array * (1. + z_array)
def recentre_nz(z_sim_edges, z_samp_centre, nz_input):
    """
    Takes input n(z) sampled at z_samp_centre
    and evaluates interpolated n(z) at new z values
    to match a simulation at z_sim_edges

    :param z_sim_edges: new z values for n(z)
    :param z_samp_centre: original z values for n(z)
    :param nz_input: original n(z)
    :return: new n(z)
    """

    nz_input = np.interp(z_sim_edges[1:],z_samp_centre, nz_input)

    return nz_input/np.sum(nz_input*(z_sim_edges[1:]-z_sim_edges[:-1]))



def E_sq(z, om0):
    """
    A function giving Hubble's law for flat cosmology

    :param z: redshift value
    :param om0: matter density
    :return: A value for the Hubble parameter
    """
    return om0 * (1 + z) ** 3 + 1 - om0


def f_integrand(z, om0):
    """
    A function for the redshift integrand in the intrinsic alignment calculation

    :param z: redshift value
    :param om0: matter density
    :return: redshift integrand
    """
    return (z + 1) / (E_sq(z, om0)) ** 1.5


def D_single(z, om0):
    """
    Provides the normalised linear growth factor

    :param z: single redshift value
    :param om0: matter density
    :return: normalised linear growth factor
    """
    first_integral = sp.integrate.quad(f_integrand, z, np.inf, args=(om0))[0]
    second_integral = sp.integrate.quad(f_integrand, 0, np.inf, args=(om0))[0]

    return (E_sq(z, om0) ** 0.5) * first_integral / second_integral


def D_1(z, om0):
    """
    Normalised linear growth factor (D_plus)

    :param z: single redshift value or array values
    :param om0: matter density
    :return: normalised linear growth factor
    """
    
    if (isinstance(z, float)) or (isinstance(z, int)):
        D_values = D_single(z, om0)
    else:
        z = list(z)
        D_values = [D_single(z[i], om0) for i in range(len(z))]
        D_values = np.array(D_values)
    
    return D_values


def F_nla(z, om0, A_ia, rho_c1, eta=0., z0=0., lbar=0., l0=1e-9, beta=0.):
    """
    NLA intrinsic alignment amplitude

    :param z: redshift value
    :param om0: matter density
    :param A_ia: amplitude parameter
    :param rho_c1: rho_crit x C1 (C1 approx 1.508e+27 cm3 / g)
    :param eta: redshift dependence
    :param z0: arbitrary redshift pivot parameter
    :param lbar: average luminosity of source galaxy population
    :param l0: arbitrary luminosity pivot parameter
    :param beta: luminosity dependence
    :return: NLA F(z) amplitude
    """
    
    prefactor = - A_ia * rho_c1 * om0 
    inverse_linear_growth = 1. / D_1(z, om0)
    redshift_dependence = ((1+z)/(1+z0))**eta
    luminosity_dependence = (lbar/l0)**beta
    
    return prefactor * inverse_linear_growth * redshift_dependence * luminosity_dependence

