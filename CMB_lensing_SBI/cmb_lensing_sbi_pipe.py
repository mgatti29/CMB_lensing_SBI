import numpy as np
import pixell
from pixell import curvedsky
import healpy as hp
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
import orphics
from orphics import io,maps



def kspace_coadd(map_alms,lbeams,noise,fkbeam=1):
    """map_alms is an array containing the coadded alms as arrays to be coadded. This is NOT beam deconvolved
       lbeams are the beam in harmonic space ordered the same way as the coadded alms in map_alms
       noise corresponds to the noise power of the coadded maps above. This is not beam deconvolved
       fkbeam is the common beam to be applied to the kspace coadd map """

    coalms=np.zeros(map_alms[0].shape)
    coalms=coalms.astype(complex)
    denom = np.sum(lbeams**2 / noise,axis=0)
    for i in range(len(noise)):
        weighted_alms=hp.almxfl(map_alms[i],lbeams[i]/noise[i])
        weighted_alms[~np.isfinite(weighted_alms)] = 0
        a=hp.almxfl(weighted_alms,1/(denom))
        a[~np.isfinite(a)] = 0
        coalms+=a
    return coalms




def mask_kspace(shape,wcs, lxcut = None, lycut = None, lmin = None, lmax = None):
    output = enmap.ones(shape[-2:],wcs, dtype = int)
    if (lmin is not None) or (lmax is not None): modlmap = enmap.modlmap(shape, wcs)
    if (lxcut is not None) or (lycut is not None): ly, lx = enmap.laxes(shape, wcs, oversample=1)
    if lmin is not None:
        output[np.where(modlmap <= lmin)] = 0
    if lmax is not None:
        output[np.where(modlmap >= lmax)] = 0
    if lxcut is not None:
        output[:,np.where(np.abs(lx) < lxcut)] = 0
    if lycut is not None:
        output[np.where(np.abs(ly) < lycut),:] = 0
    return output



def coadd_mapnew(map_list,ivar_list,a):
    """return coadded map from splits, the map in maplist contains I,Q,U 
    a=0,1,2 selects one of I Q U """
    wcs=map_list[0].wcs
    map_list=np.array(map_list)
    ivar_list=np.array(ivar_list)
    coadd_map= np.sum(map_list[:,a] * ivar_list, axis = 0)
    #coadd_map/=((np.sum(ivar_list*mask, axis = 0)))
    coadd_map/=((np.sum(ivar_list, axis = 0)))
    print('ignore warning: some ivars are 0 but we are taking this into account ')
    #coadd_map/=((np.sum(ivar_list, axis = 0)))
    coadd_map[~np.isfinite(coadd_map)] = 0
    coadd_map = enmap.ndmap(coadd_map,wcs)
    return coadd_map    

def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def get_datanoise(map_list,ivar_list, a, b, mask,beam,N=20,beam_deconvolve=True,lmax=6000):
    ### THIS IS A FUNCTION FROM SOLENSPIPE UTILITY WITH MINOR MODIFICATIONS
    """
    Calculate the noise power of a coadded map given a list of maps and list of ivars.
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    a: 0,1,2 for I,Q,U respectively
    b:0,1,2 for I,Q,U, respectively
    N: window to smooth the power spectrum by in the rolling average.
    mask: apodizing mask

    Output:
    1D power spectrum accounted for w2 from 0 to 10000
    """
    
    pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)


    cl_ab=[]
    n = len(map_list)
    #calculate the coadd maps
    if a!=b:
        coadd_a=coadd_mapnew(map_list,ivar_list,a)
        coadd_b=coadd_mapnew(map_list,ivar_list,b)
    else:
        coadd_a=coadd_mapnew(map_list,ivar_list,a)

    for i in range(n):
        print(i)
        if a!=b:
            d_a=map_list[i][a]-coadd_a
            #noise_a=d_a*mask
            noise_a=d_a #noise already masked
            alm_a=curvedsky.map2alm(noise_a,lmax=lmax)
            d_b=map_list[i][b]-coadd_b
            noise_b=d_b
            alm_b=curvedsky.map2alm(noise_b,lmax=lmax)
            cls = hp.alm2cl(alm_a,alm_b)
            cl_ab.append(cls)
        else:
            d_a=map_list[i][a]-coadd_a

            noise_a=d_a
            #enplot.write(f"/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/d_{a}_0",enplot.plot(noise_a))    
            print("generating alms")
            alm_a=curvedsky.map2alm(noise_a,lmax=lmax)
            alm_a=alm_a.astype(np.complex128)
            if beam_deconvolve:
                alm_a = curvedsky.almxfl(alm_a,lambda x: 1/beam(x)) 
            cls = hp.alm2cl(alm_a)
            cl_ab.append(cls)
    cl_ab=np.array(cl_ab)
    #sqrt_ivar=np.sqrt(ivar_eff(0,ivar_list))

    mask=mask
    mask[mask<=0]=0
    w2=np.sum((mask**2)*pmap) /np.pi / 4.
    print(w2)
    if n == 1:  #MY MODIFICATION FOR A 1-SPLIT RUN
        power = np.sum(cl_ab, axis=0)
    else:
        power = 1/n/(n-1) * np.sum(cl_ab, axis=0)
    ls=np.arange((power.shape[-1]))
    power[~np.isfinite(power)] = 0
    power=rolling_average(power, N)
    bins=np.arange(len(power))
    power=maps.interp(bins,power)(ls)

    return power / w2

def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = orphics.stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return cents,bls



def reconvolve_maps(maps,mask,beamdec,beamconv,lmax=6000):
    "deconvolve the beam of a map and return a map convolved with new beam"
    shape=maps.shape
    wcs=maps.wcs
    alm_a=curvedsky.map2alm(maps*mask,lmax=lmax)
    alm_a = curvedsky.almxfl(alm_a,lambda x: 1/beamdec(x)) 
    convolved_alm=curvedsky.almxfl(alm_a,lambda x: beamconv(x)) 
    reconvolved_map=curvedsky.alm2map(convolved_alm,enmap.empty(shape,wcs))
    return reconvolved_map

def deconvolve_maps(maps,mask,beam,lmax=6000):
    "deconvolve the beam of a map" 
    "function from solenspipe.utility (simgen) but slightly modified mask application"
    shape=maps.shape
    wcs=maps.wcs
    maps[:,mask<0.25]=0
    alm_a=curvedsky.map2alm(maps,lmax=lmax)
    alm_a = curvedsky.almxfl(alm_a,lambda x: 1/beam(x)) 
    reconvolved_map=curvedsky.alm2map(alm_a,enmap.empty(shape,wcs))
    return reconvolved_map

def kspace_mask(imap, vk_mask=[-90,90], hk_mask=[-50,50], normalize="phys", deconvolve=False):

    """Filter the map in Fourier space removing modes in a horizontal and vertical band
    defined by hk_mask and vk_mask. This is a faster version that what is implemented in pspy
    We also include an option for removing the pixel window function. Stolen from Will C who stole it from PS group.
    
    Parameters
    ---------
    imap: ``so_map``
        the map to be filtered
    vk_mask: list with 2 elements
        format is fourier modes [-lx,+lx]
    hk_mask: list with 2 elements
        format is fourier modes [-ly,+ly]
    normalize: string
        optional normalisation of the Fourier transform
    inv_pixwin_lxly: 2d array
        the inverse of the pixel window function in fourier space
    """
    if vk_mask is None and hk_mask is None:
        imap=imap
        if deconvolve:
            pow=-1
            wy, wx = enmap.calc_window(imap.shape)
            ft = enmap.fft(imap, normalize=normalize)
            ft = ft* wy[:,None]**pow * wx[None,:]**pow
            
        imap[:,:] = np.real(enmap.ifft(ft, normalize=normalize))
        return imap
    lymap, lxmap = imap.lmap()
    ly, lx = lymap[:,0], lxmap[0,:]

   # filtered_map = map.copy()
    ft = enmap.fft(imap, normalize=normalize)
    
    if vk_mask is not None:
        id_vk = np.where((lx > vk_mask[0]) & (lx < vk_mask[1]))
    if hk_mask is not None:
        id_hk = np.where((ly > hk_mask[0]) & (ly < hk_mask[1]))

    ft[...,: , id_vk] = 0.
    ft[...,id_hk,:]   = 0.

    if deconvolve:
        pow=-1
        wy, wx = enmap.calc_window(imap.shape)
        ft = ft* wy[:,None]**pow * wx[None,:]**pow
        
    imap[:,:] = np.real(enmap.ifft(ft, normalize=normalize))
    return imap

def get_nalms(lmax, mmax = None):
    '''
    Calculate number of alms given (ell max, m max) [healpy format]
    '''

    if mmax is None:
        mmax = lmax # m = ell
    return int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)



def rand_alm_(ps, ainfo=None, lmax=None, seed=None, dtype=np.complex128, m_major=True, return_ainfo=False):
    """
    Generate a random alm (spherical harmonic coefficients) from a given power spectrum.

    Parameters:
    ps           : Power spectrum array.
    ainfo        : Information about the spherical harmonics.
    lmax         : Maximum multipole moment.
    seed         : Seed for the random number generator.
    dtype        : Data type for the coefficients, default is np.complex128.
    m_major      : Flag to specify memory layout (major order).
    return_ainfo : Flag to determine if ainfo should be returned.

    Returns:
    alm          : Generated random alm.
    ainfo        : (Optional) Spherical harmonics information.
    """

    # Determine the real data type corresponding to the complex dtype
    rtype = np.zeros([0], dtype=dtype).real.dtype
    
    # Prepare power spectrum and spherical harmonics information
    wps, ainfo = curvedsky.prepare_ps(ps, ainfo=ainfo, lmax=lmax)
    
    # Generate white noise alm
    alm = curvedsky.rand_alm_white(ainfo, pre=[wps.shape[0]], seed=seed, dtype=dtype, m_major=m_major)
    
    # Compute the square root of the power spectrum
    ps12 = curvedsky.enmap.multi_pow(wps, 0.5)
    
    # Scale alm by the power spectrum
    ainfo.lmul(alm, (ps12 / 2**0.5).astype(rtype, copy=False), alm)
    
    # Ensure that the real part of alm is zero for m=0
    alm[:, :ainfo.lmax + 1].imag = 0
    
    # Scale the real part of alm for m=0
    alm[:, :ainfo.lmax + 1].real *= 2**0.5
    
    # If power spectrum has only one dimension, reduce the dimension of alm
    if ps.ndim == 1:
        alm = alm[0]
    
    # Return alm and optionally ainfo
    if return_ainfo:
        return alm, ainfo
    else:
        return alm
    
    
def gauss_beam(ell, fwhm):
    """
    Calculates a Gaussian beam in Fourier space.

    Parameters:
    ell (ndarray): Array of multipole moments.
    fwhm (float): Full width at half maximum (FWHM) of the beam in arcminutes.

    Returns:
    ndarray: Gaussian beam values for the given multipole moments.
    """
    # Convert FWHM from arcminutes to radians
    tht_fwhm = np.deg2rad(fwhm / 60.)
    
    # Calculate and return the Gaussian beam values
    return np.exp(-(tht_fwhm**2.) * (ell**2.) / (16. * np.log(2.)))

def white_noise(shape, wcs, seed=None, div=None):
    """
    Generates white noise with optional division and random seed.

    Parameters:
    shape (tuple): Shape of the output array.
    wcs: World Coordinate System information (unused in this function).
    seed (int, optional): Seed for the random number generator.
    div (float, optional): Divisor for scaling the noise.

    Returns:
    ndarray: Array containing the generated white noise.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate white noise and scale it if a divisor is provided
    return np.random.standard_normal(shape) / np.sqrt(div)


def gapfill_edge_conv_flat(map, mask, ivar=None, alpha=-3, edge_rad=1*utils.arcmin, rmin=2*utils.arcmin, tol=1e-8):
    """Gapfill by doing a masked convolution with a profile that
    prioritizes nearby areas but still includes further ones.
    The mask should be 1 in bad regions and 0 in good regions.
    
    This version assumes a flat sky. This helps not only with speed (FFTs vs. SHTs),
    but also with numerical stability. The cost is that the gapfilling gets
    a bit elliptical away from the equator, but that probably isn't a big issue
    in practice, since the gapfilling is only a rough approximation in the
    first place.
    
    This method becomes numerically unstable when r**alpha becomes too
    small. For my test case with 80 arcmin holes, alpha=-3 works while
    alpha = -5 start breaking down. tol helps this happen more gracefully.
    The inpainting should be valid up to a radius of tol**(1/alpha)*rmin
    from the hole edge. For the default alpha=-3, rmin=2 and tol=1e-8, this
    gives 15 degrees, which is more than enough for typical gapfilling."""
    refpix = np.array(map.shape[-2:])//2
    rmax   = tol**(1/alpha) * rmin
    r      = enmap.shift(map.distance_from(map.pix2sky(refpix)[:,None],rmax=rmax).astype(map.dtype),-refpix,keepwcs=True)
    r      = np.maximum(r, rmin)
    rprof  = (r/utils.arcmin)**alpha
    del r
    lprof = enmap.fft(rprof)
    del rprof
    # Build the weight. This is the edge of the mask
    edist  = (1-mask).distance_transform(rmax=edge_rad).astype(map.dtype)
    weight = ((edist>0)&(edist<edge_rad))
    del edist
    # Do the masked convolution
    def conv(lprof,map): return enmap.ifft(lprof*enmap.fft(map)).real
    rhs   = conv(lprof, weight*map)
    div   = conv(lprof, weight)
    del weight, lprof
    div   = np.maximum(div,np.max(div)*(tol*100))
    omap  = rhs/div
    del rhs, div
    # Restore known part
    omap[...,~mask] = map[...,~mask]
    # Add noise
    if not(ivar is None):
        n = white_noise(omap.shape,omap.wcs,div=ivar)
        omap[...,mask] = omap[...,mask] + n[...,mask]
    return omap.astype(map.dtype)