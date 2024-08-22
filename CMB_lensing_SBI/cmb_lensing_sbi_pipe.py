import numpy as np
import pixell
from pixell import curvedsky
import healpy as hp
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
import orphics
from orphics import io,maps
import pytempura
from falafel import qe


'''
These are a number of routines I took from solenspipe or Karen's lensing utils.
'''


# https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/solenspipe.py#L106
def split_phi_to_cl(xy,uv,m=4,cross=False,ikalm=None):
    phi_x=xy[0];phi01=xy[1];phi02=xy[2];phi03=xy[3];phi12=xy[4];phi13=xy[5];phi23=xy[6];phi_x0=xy[7];phi_x1=xy[8];phi_x2=xy[9];phi_x3=xy[10]
    phi_xp=uv[0];phi01p=uv[1];phi02p=uv[2];phi03p=uv[3];phi12p=uv[4];phi13p=uv[5];phi23p=uv[6];phi_x0p=uv[7];phi_x1p=uv[8];phi_x2p=uv[9];phi_x3p=uv[10]
    if cross is False:
        tg1=m**4*curvedsky.alm2cl(phi_x,phi_xp)
        tg2=-4*m**2*(curvedsky.alm2cl(phi_x0,phi_x0p)+curvedsky.alm2cl(phi_x1,phi_x1p)+curvedsky.alm2cl(phi_x2,phi_x2p)+curvedsky.alm2cl(phi_x3,phi_x3p))
        tg3=4*(curvedsky.alm2cl(phi01,phi01p)+curvedsky.alm2cl(phi02,phi02p)+curvedsky.alm2cl(phi03,phi03p)+curvedsky.alm2cl(phi12,phi12p)+curvedsky.alm2cl(phi13,phi13p)+cs.alm2cl(phi23,phi23p))
    else:
        tg1=m**4*curvedsky.alm2cl(phi_x,ikalm)
        tg2=-4*m**2*(curvedsky.alm2cl(phi_x0,ikalm)+curvedsky.alm2cl(phi_x1,ikalm)+curvedsky.alm2cl(phi_x2,ikalm)+curvedsky.alm2cl(phi_x3,ikalm))
        tg3=4*(curvedsky.alm2cl(phi01,ikalm)+curvedsky.alm2cl(phi02,ikalm)+curvedsky.alm2cl(phi03,ikalm)+curvedsky.alm2cl(phi12,ikalm)+curvedsky.alm2cl(phi13,ikalm)+curvedsky.alm2cl(phi23,ikalm))

    auto =(1/(m*(m-1)*(m-2)*(m-3)))*(tg1+tg2+tg3)
    return auto

#https://github.com/mgatti29/CMB_lensing_SBI/blob/lensing_pipe/code/lensing_pipeline/utils.py#L143
def phi_to_cl(xy,uv,m=1,cross=False,ikalm=None):
    if cross:
        cl = curvedsky.alm2cl(xy[0],ikalm)
    else:
        cl = curvedsky.alm2cl(xy[0],uv[0])
    return cl

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/solenspipe.py#L23
def four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0=None,Xdatp_1=None,Xdatp_2=None,Xdatp_3=None,q_func1=None):
    """Return kappa_alms combinations required for the 4cross estimator in Eq. 38 of arXiv:2011.02475v1 .

    Args:
        Xdat_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0
        Xdat_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1
        Xdat_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2
        Xdat_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3
        q_func1 (function): function for quadratic estimator
        Xdatp_0 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 0 used for RDN0 for different sim data combination
        Xdatp_1 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 1 used for RDN0 for different sim data combination
        Xdatp_2 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 2 used for RDN0 for different sim data combination
        Xdatp_3 (array): [fTalm,fEalm,fBalm] list of filtered alms from split 3 used for RDN0 for different sim data combination
        qfunc2 ([type], optional): [description]. Defaults to None.

    Returns:
        array: Combination of reconstructed kappa alms
    """
    q_bh_1=q_func1
    if Xdatp_0 is None:
        print("none")
        
        phi_xy00 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_0))
        phi_xy11 = plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_1))
        phi_xy22 = plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_2))
        phi_xy33 = plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_3))
        phi_xy01 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_1))+plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_0)))
        phi_xy02 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_0)))
        phi_xy03 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdat_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_0)))
        phi_xy10=phi_xy01
        phi_xy12= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_1)))
        phi_xy13= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdat_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_1)))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdat_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdat_2)))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4
    
    else:
       
        phi_xy00 = plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_0))
        phi_xy11 = plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_1))
        phi_xy22 = plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_2))
        phi_xy33 = plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_3))
        phi_xy01 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_1))+plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_0)))
        phi_xy02 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_0)))
        phi_xy03 = 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_0,Xdatp_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_0)))
        phi_xy10=phi_xy01
        phi_xy12= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_2))+plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_1)))
        phi_xy13= 0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_1,Xdatp_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_1)))
        phi_xy20=phi_xy02
        phi_xy21=phi_xy12
        phi_xy23=0.5*(plensing.phi_to_kappa(q_bh_1(Xdat_2,Xdatp_3))+plensing.phi_to_kappa(q_bh_1(Xdat_3,Xdatp_2)))
        phi_xy30=phi_xy03
        phi_xy31=phi_xy13
        phi_xy32=phi_xy23
        phi_xy_hat=(phi_xy00+phi_xy11+phi_xy22+phi_xy33+phi_xy01+phi_xy02+phi_xy03+phi_xy10+phi_xy12+phi_xy13+phi_xy20+phi_xy21+phi_xy23+phi_xy30+phi_xy31+phi_xy32)/4**2
        phi_xy_X=phi_xy_hat-(phi_xy00+phi_xy11+phi_xy22+phi_xy33)/4**2                        
        phi_xy0=(phi_xy00+phi_xy01+phi_xy02+phi_xy03)/4
        phi_xy1=(phi_xy10+phi_xy11+phi_xy12+phi_xy13)/4
        phi_xy2=(phi_xy20+phi_xy21+phi_xy22+phi_xy23)/4
        phi_xy3=(phi_xy30+phi_xy31+phi_xy32+phi_xy33)/4
        phi_xy_x0=phi_xy0-phi_xy00/4
        phi_xy_x1=phi_xy1-phi_xy11/4
        phi_xy_x2=phi_xy2-phi_xy22/4
        phi_xy_x3=phi_xy3-phi_xy33/4

    phi_xy=np.array([phi_xy_X,phi_xy01,phi_xy02,phi_xy03,phi_xy12,phi_xy13,phi_xy23,phi_xy_x0,phi_xy_x1,phi_xy_x2,phi_xy_x3])
    

    return phi_xy

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/solenspipe.py#L230
def get_qfunc(px,ucls,mlmax,est1,Al1=None,est2=None,Al2=None,Al3=None,R12=None,profile=None):
    """
    Prepares a qfunc lambda function for an estimator est1. Optionally,
    normalize it with Al1. Optionally, bias harden it (which
    results in a normalized estimator) against est2 with
    normalization Al2 and unnormalized cross-response R12.


    Parameters
    ----------

    px : object
        A falafal.qe.pixelization object that holds healpix or rectangular
        pixel information and associated common functions
    ucls : dict
        A dictionary mapping TT,TE,EE,BB to spectra used in the response
        of various estimators. Typically these are gradient-field spectra
        or lensed field spectra.
    mlmax : int
        Maximum multipole for alm transforms
    est1 : str
        The name of a pre-defined falafel estimator. e.g. MV,MVPOL,TT,
        EB,TE,EE,TB.
    Al1 : ndarray
        A (2,mlmax) shape numpy array containing the gradient-like (e.g. lensing
        potential) and curl-like normalization.
    est2 : str, optional
        The name of a pre-defined falafel estimator to bias harden against
    Al2 : ndarray, optional
        A (mlmax,) shape numpy array containing the normalization of the 
        estimator being hardened against.
    Al3 : ndarray, optional
        A (mlmax,) shape numpy array containing the normalization of the 
        TT estimator used when calculating BH estimator.
    R12 : ndarray, optional
        An (mlmax,) or (1,mlmax) or (2,mlmax) shape numpy array containing 
        the unnormalized cross-response of est1 and est2. If two components
        are present, then the curl of est1 is also bias hardened using the
        cross-response of est2 with curl specified through the second
        component.
    profile : (mlmax) array, default=None
        An array to use as the profile for profile-hardening, when est2="SRC".
        If not provided, will just do point-source hardening. 

    Returns
    -------
    qfunc : function
        Quadratic estimator lambda function
    
    """
    est1 = est1.upper()
    assert est1 in pytempura.est_list
    if Al1 is not None:
        assert Al1.ndim==2, "Both gradient and curl normalizations need to be present."
    if est2 is not None:
        bh = True
        assert est2 in pytempura.est_list
        assert Al1 is not None
        assert Al2 is not None
        if Al2.ndim==2:
            assert Al2.shape[0]==1
            Al2 = Al2[0]
        else:
            assert Al2.ndim==1
        assert R12 is not None
        if R12.ndim==1: 
            R12 = R12[None]
        else: 
            assert R12.ndim==2
    else:
        bh = False

    assert est1 in ['TT','TE','EE','EB','TB','MV','MVPOL','SHEAR'] # TODO: add other
    if est1=='SHEAR':
        qfunc1 = lambda X,Y: qe.qe_shear(px,mlmax,
                            Talm=X[0],fTalm=Y[1])
    else:
        qfunc1 = lambda X,Y: qe.qe_all(px,ucls,mlmax,
                                    fTalm=Y[0],fEalm=Y[1],fBalm=Y[2],
                                    estimators=[est1],
                                    xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])[est1]

    if bh:
        assert est2 in ['SRC','MASK'] # TODO: add mask
        if est2 == 'SRC':
            qfunc2 = lambda X,Y: qe.qe_source(px,mlmax,Y[0],profile=profile,xfTalm=X[0])
        elif est2 == 'mask':
            qfunc2 = lambda X,Y: qe.qe_mask(px,ucls,mlmax,fTalm=Y[0],xfTalm=X[0])
        # The bias-hardened estimator Eq 27 of arxiv:1209.0091
        if R12.shape[0]==1:

            if est1=='TT':
                # Bias harden only gradient e.g. source hardening
                def retfunc(X,Y):
                    q1 = qfunc1(X,Y)
                    q2 = qfunc2(X,Y)
                    g = curvedsky.almxfl( \
                                (curvedsky.almxfl(q1[0],Al1[0]) - \
                                    curvedsky.almxfl(qfunc2(X,Y),Al1[0] * Al2 * R12[0])) , \
                                1. / (1. - Al1[0] * Al2 * R12[0]**2.) \
                    )
                    c = curvedsky.almxfl(q1[1],Al1[1])
                    return np.asarray((g,c))
            else:
                def retfunc(X,Y):
                    print('test bh MV')
                    qfuncTT= lambda X,Y: qe.qe_all(px,ucls,mlmax,
                                        fTalm=Y[0],fEalm=Y[1],fBalm=Y[2],
                                        estimators=['TT'],
                                        xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])['TT']
                    q1=qfuncTT(X,Y)

                    q2 = qfunc2(X,Y)

                    qfuncmv=lambda X,Y: qe.qe_all(px,ucls,mlmax,
                                        fTalm=Y[0],fEalm=Y[1],fBalm=Y[2],
                                        estimators=['MV'],
                                        xfTalm=X[0],xfEalm=X[1],xfBalm=X[2])['MV']
                    
                    qmv=qfuncmv(X,Y)
                    g_bh_TT = curvedsky.almxfl( \
                                (curvedsky.almxfl(q1[0],Al3[0]) - \
                                    curvedsky.almxfl(qfunc2(X,Y),Al3[0] * Al2 * R12[0])) , \
                                1. / (1. - Al3[0] * Al2 * R12[0]**2.) \
                    )
                    g= curvedsky.almxfl(qmv[0]-q1[0]+curvedsky.almxfl(g_bh_TT,1/Al3[0]),Al1[0])
                    c = curvedsky.almxfl(qmv[1],Al1[1])


                    return np.asarray((g,c))

        elif R12.shape[0]==2:
            # Bias harden both e.g. mask hardening
            def retfunc(X,Y):
                q1 = qfunc1(X,Y)
                q2 = qfunc2(X,Y)
                g = curvedsky.almxfl( \
                               (curvedsky.almxfl(q1[0],Al1[0]) - \
                                curvedsky.almxfl(qfunc2(X,Y),Al1[0] * Al2 * R12[0])) , \
                               1. / (1. - Al1[0] * Al2 * R12[0]**2.) \
                )
                c = curvedsky.almxfl( \
                               (curvedsky.almxfl(q1[1],Al1[1]) - \
                                curvedsky.almxfl(qfunc2(X,Y),Al1[1] * Al2 * R12[1])) , \
                               1. / (1. - Al1[1] * Al2 * R12[1]**2.) \
                )
                return np.asarray((g,c))

        return retfunc
                
    else:
        if Al1 is not None: 
            # TODO: Improve this construct by building a multi-dimensional almxfl
            def retfunc(X,Y):
                recon = qfunc1(X,Y)
                return np.asarray((curvedsky.almxfl(recon[0],Al1[0]),curvedsky.almxfl(recon[1],Al1[1])))
            return retfunc
        else: return qfunc1

    
#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L809
def w_n(mask,n):
    """wrapper for solenspipe's wfactor function"""
    pmap = enmap.pixsizemap(mask.shape,mask.wcs)
    return maps.wfactor(n,mask,sht=True,pmap=pmap)

#https://github.com/mgatti29/CMB_lensing_SBI/blob/lensing_pipe/code/lensing_pipeline/utils.py#L103
def smooth_pack(alms,mask,n):
    cltt = smooth_cls(hp.alm2cl(alms[0])/w_n(mask,n))
    clee=smooth_cls(hp.alm2cl(alms[1])/w_n(mask,n)) #this is signal+noise
    clbb=smooth_cls(hp.alm2cl(alms[2])/w_n(mask,n))
    clte=smooth_cls(hp.alm2cl(alms[0],alms[1])/w_n(mask,n))
    return np.array([cltt,clee,clbb,clte])

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L655
def smooth_cls(cl,points=300):
    """bin and interpolate a cl to smooth it"""
    bin_edges = np.linspace(2,len(cl),points).astype(int)
    cents,cls=bandedcls(cl,bin_edges)
    cls=maps.interp(cents,cls)(np.arange(len(cl)))
    return cls


#https://github.com/mgatti29/CMB_lensing_SBI/blob/lensing_pipe/code/lensing_pipeline/utils.py#L110
def reshape_alm(fname, mask, lmax):
    # Read spherical harmonic coefficients (alm) from the file 'fname'.
    # The coefficients are stored in the first, second, and third HDU (Header/Data Units).
    alm = hp.read_alm(fname, hdu=(1, 2, 3))
    
    # Convert the spherical harmonic coefficients (alm) into a pixel-space map (pmap).
    # 'enmap.empty((3,)+mask.shape,mask.wcs)' creates an empty map with the same shape and WCS (World Coordinate System) as the mask.
    # The resulting map is then multiplied by the mask to apply the mask to the pixel-space map.
    pmap = curvedsky.alm2map(alm, enmap.empty((3,) + mask.shape, mask.wcs)) * mask
    
    # Convert the pixel-space map back into spherical harmonic coefficients (alm).
    # 'lmax' specifies the maximum multipole order for the transformation.
    oalms = curvedsky.map2alm(pmap, lmax=lmax)
    
    # Replace any non-finite values (e.g., NaN or infinity) in the output alms with zero.
    oalms[~np.isfinite(oalms)] = 0
    
    # Ensure the output alms are of type 'complex128' (complex numbers with double precision).
    oalms = oalms.astype(np.complex128)
    
    # Return the reshaped spherical harmonic coefficients (alm).
    return oalms


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


#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L119
def coadd_mapnew(map_list,ivar_list,a):
    """return coadded map from splits, the map in maplist contains I,Q,U 
    a=0,1,2 selects one of I Q U """
    wcs=map_list[0].wcs
    map_list=np.array(map_list)
    ivar_list=np.array(ivar_list)
    coadd_map= np.sum(map_list[:,a] * ivar_list[:,a], axis = 0)
    #coadd_map/=((np.sum(ivar_list*mask, axis = 0)))
    coadd_map/=((np.sum(ivar_list[:,a], axis = 0)))
    print('ignore warning: some ivars are 0 but we are taking this into account ')
    #coadd_map/=((np.sum(ivar_list, axis = 0)))
    coadd_map[~np.isfinite(coadd_map)] = 0
    coadd_map = enmap.ndmap(coadd_map,wcs)
    return coadd_map    

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L105
def rolling_average(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L233
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

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L613
def bandedcls(cl,_bin_edges):
    ls=np.arange(cl.size)
    binner = orphics.stats.bin1D(_bin_edges)
    cents,bls = binner.bin(ls,cl)
    return cents,bls


#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L575
def reconvolve_maps(maps,mask,beamdec,beamconv,lmax=6000):
    "deconvolve the beam of a map and return a map convolved with new beam"
    shape=maps.shape
    wcs=maps.wcs
    alm_a=curvedsky.map2alm(maps*mask,lmax=lmax)
    alm_a = curvedsky.almxfl(alm_a,lambda x: 1/beamdec(x)) 
    convolved_alm=curvedsky.almxfl(alm_a,lambda x: beamconv(x)) 
    reconvolved_map=curvedsky.alm2map(convolved_alm,enmap.empty(shape,wcs))
    return reconvolved_map

#https://github.com/simonsobs/so-lenspipe/blob/a949e865a93569cb7a1a05f525aeff7e65c7d7a4/solenspipe/utility.py#L566
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

#https://github.com/mgatti29/CMB_lensing_SBI/blob/lensing_pipe/code/lensing_pipeline/utils.py#L23
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
    
#https://github.com/mgatti29/CMB_lensing_SBI/blob/lensing_pipe/code/lensing_pipeline/utils.py#L18
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