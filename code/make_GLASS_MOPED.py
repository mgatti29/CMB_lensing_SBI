import glass
import astropy.io.fits as fits
import numpy as np
import camb
from cosmology import Cosmology
from glass import fields
import glass.shells
import frogress
from glass import lensing
import timeit
from CMB_lensing_SBI.healpy_utils import *
from CMB_lensing_SBI.Theory_camb import theory,cosmo
#import glass.ext.camb
from bornraytrace import intrinsic_alignments as iaa
import gc
import os
import random
def camb_tophat_weight(z):
    '''Weight function for tophat window functions and CAMB.

    This weight function linearly ramps up the redshift at low values,
    from :math:`w(z = 0) = 0` to :math:`w(z = 0.1) = 1`.

    '''
    return np.clip(z/0.1, None, 1.)

def matter_cls(pars, lmax, ws, *, limber=False, limber_lmin=100):
    '''Compute angular matter power spectra using CAMB.'''

    # make a copy of input parameters so we can set the things we need
    pars = pars.copy()

    # set up parameters for angular power spectra
    pars.WantTransfer = False
    pars.WantCls = True
    pars.Want_CMB = False
    pars.min_l = 1
    pars.set_for_lmax(lmax)

    # set up parameters to only compute the intrinsic matter cls
    pars.SourceTerms.limber_windows = limber
    pars.SourceTerms.limber_phi_lmin = limber_lmin
    pars.SourceTerms.counts_density = True
    pars.SourceTerms.counts_redshift = False
    pars.SourceTerms.counts_lensing = False
    pars.SourceTerms.counts_velocity = False
    pars.SourceTerms.counts_radial = False
    pars.SourceTerms.counts_timedelay = False
    pars.SourceTerms.counts_ISW = False
    pars.SourceTerms.counts_potential = False
    pars.SourceTerms.counts_evolve = False

    sources = []
    for za, wa, _ in ws:
        s = camb.sources.SplinedSourceWindow(z=za, W=wa)
        sources.append(s)
    pars.SourceWindows = sources

    n = len(sources)
    cls = camb.get_results(pars).get_source_cls_dict(lmax=lmax, raw_cl=True)

    for i in range(1, n+1):
        if np.any(cls[f'W{i}xW{i}'] < 0):
            warnings.warn('negative auto-correlation in shell {i}; improve accuracy?')

    return [cls[f'W{i}xW{j}'] for i in range(1, n+1) for j in range(i, 0, -1)]

import pickle
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        mute =  pickle.load(f)
        f.close()
    return mute

def addSourceEllipticity(self,es, es_colnames=("e1","e2"), rs_correction=True, inplace=False):
    """
    Adds intrinsic source ellipticity to shear measurements.

    Args:
        es (array): Array of intrinsic ellipticities.
        es_colnames (tuple, optional): Column names for the intrinsic ellipticities. Defaults to ("e1", "e2").
        rs_correction (bool, optional): Flag indicating whether to apply Rousseeuw and Schneider correction. Defaults to True.
        inplace (bool, optional): Flag indicating whether to modify the input shear measurements in-place. Defaults to False.

    Returns:
        tuple: Modified shear measurements.
    """
    assert len(self) == len(es)
    es_c = np.array(es[es_colnames[0]] + es[es_colnames[1]] * 1j)
    g = np.array(self["shear1"] + self["shear2"] * 1j)
    e = es_c + g
    if rs_correction:
        e /= (1 + g.conjugate() * es_c)
    if inplace:
        self["shear1"] = e.real
        self["shear2"] = e.imag
    else:
        return (e.real, e.imag)

def gk_inv(K,KB,nside,lmax):

    alms = hp.map2alm(K, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsE = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsE[ell == 0] = 0.0

    
    alms = hp.map2alm(KB, lmax=lmax, pol=False)  # Spin transform!

    ell, emm = hp.Alm.getlm(lmax=lmax)

    kalmsB = alms/( 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5)
   
    kalmsB[ell == 0] = 0.0

    _,e1t,e2t = hp.alm2map([kalmsE,kalmsE,kalmsB] , nside=nside, lmax=lmax, pol=True)
    return e1t,e2t# ,r






# cosmology for the simulation -------
'''
h = 0.6736
Ob = 0.0493
Om = 0.26
A_IA = 0
E_IA = 0
w0 = -1.
sigma_8 = 0.84
ns=0.9649
mnu=0.02



params_dict = dict()
params_dict['Om'] = Om
params_dict['Ob'] = Ob
params_dict['ns'] = ns
params_dict['sigma_8'] = sigma_8
params_dict['w'] = w0
params_dict['A_IA'] = A_IA
params_dict['E_IA'] = E_IA
params_dict['mnu'] = mnu
'''

def make_mock(params_dict, fiducial_cosmo = False, no_noise=False):
    
    Om = params_dict['Om'] 
    Ob = params_dict['Ob']
    ns = params_dict['ns']
    sigma_8 = params_dict['sigma_8']
    w0 = params_dict['w'] 
    A_IA = params_dict['A_IA'] 
    E_IA = params_dict['E_IA'] 
    mnu = params_dict['mnu']
    h = params_dict['h']
    
    if fiducial_cosmo:    
        h = 0.6736
        Ob = 0.0493
        Om = 0.26
        A_IA = 0
        E_IA = 0
        w0 = -1.
        sigma_8 = 0.84
        ns=0.9649
        mnu=0.06


    omk=0 
    tau=0.0 
    Cosmo_ = cosmo(H0=h*100, ombh2=Ob*h**2, omch2=(Om-Ob)*h**2,As = 2e-9,ns=ns,mnu=mnu,num_massive_neutrinos=3 )

    # basic parameters of the simulation
    lmax = 512*3
    nside = 512

    st = timeit.default_timer()
    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(H0=100*h, omch2=(Om-Ob)*h**2, ombh2=Ob*h**2,
                           NonLinear=camb.model.NonLinear_both,mnu=mnu,num_massive_neutrinos=3 )




    pars.DarkEnergy.set_params(w=w0) 
    pars.InitPower.set_params(As=2e-9 , ns=ns, r=0)

    pars.set_matter_power(redshifts = [0.], kmax = 2.0)
    results_ = camb.get_results(pars)
    r0 = (sigma_8**2/ results_.get_sigma8()**2)
    pars.InitPower.set_params(As=2e-9*r0, ns=ns, r=0)





    # get the cosmology from CAMB
    cosmo_glass = Cosmology.from_camb(pars)

    # shells of 200 Mpc in comoving distance spacing
    zb = glass.shells.distance_grid(cosmo_glass, 0.0001, 2.5, dx=200)

    print (len(zb))

    # add CMB lensing map here ~

    # uniform matter weight function
    # CAMB requires linear ramp for low redshifts

    ws = glass.shells.tophat_windows(zb, weight=camb_tophat_weight)

    # compute angular matter power spectra with CAMB
    cls = matter_cls(pars, lmax, ws)

    end = timeit.default_timer()

    print ('time needed ',end-st)
    
    

    # Redshift distributions -------------------------------------------------------------------------------------------

    file_2pt =  '//global/cfs/cdirs//des/www/y3_chains/data_vectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits'

    mu = fits.open(file_2pt)
    random_rel = np.random.randint(0,6000,1)[0]
    redshift_distributions_sources = {'z':None,'bins':dict()}
    redshift_distributions_sources['z'] = np.hstack([0,mu[6].data['Z_MID']])
    for ix in [1,2,3,4]:
       # #redshift_distributions_sources['bins'][ix] =  np.hstack([0,mu[8+random_rel].data['BIN{0}'.format(ix)]])
        redshift_distributions_sources['bins'][ix] = np.hstack([0,mu[6].data['BIN{0}'.format(ix)]])


    for ix in [1,2,3,4]:
        redshift_distributions_sources['bins'][ix][250:] = 0.


    mu = None
    
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM,wCDM
    from bornraytrace import intrinsic_alignments as iaa

    cosmology = wCDM(H0= h,
                 Om0=Om,#mega_fld,
                 Ode0=1-Om,#Omega_fld,
                 w0=w0)

    c1 = (5e-14 * (u.Mpc**3.)/(u.solMass * u.littleh**2) ) 
    c1_cgs = (c1* ((u.littleh/(cosmology.H0.value/100))**2.)).cgs
    rho_c1 = (c1_cgs*cosmology.critical_density(0)).value




    
    # compute Gaussian cls for lognormal fields for 3 correlated shells
    # putting nside here means that the HEALPix pixel window function is applied
    gls = fields.lognormal_gls(cls, nside=nside, lmax=lmax, ncorr=3)
    # generator for lognormal matter fields
    matter = fields.generate_lognormal(gls, nside, ncorr=3)
    # this will compute the convergence field iteratively
    convergence = lensing.MultiPlaneConvergence(cosmo_glass)

    # the integrated convergence and shear field over the redshift distribution
    kappa_bar = np.zeros((4,12*nside**2))
    gamm1_bar = np.zeros((4,12*nside**2))
    gamm2_bar = np.zeros((4,12*nside**2))
    delta_bar = np.zeros((4,12*nside**2))


    # main loop to simulate the matter fields iterative
    for i, delta_i in enumerate(matter):
        # add lensing plane from the window function of this shell
        convergence.add_window(delta_i, ws[i])

        # get convergence field
        kappa_i = convergence.kappa

        # compute shear field
        gamm1_i, gamm2_i = glass.lensing.shear_from_convergence(kappa_i)

        # get the restriction of the dndz to this shell
        for ix in [1,2,3,4]:
            z_i, dndz_i = glass.shells.restrict(redshift_distributions_sources['z']+params_dict['dz{0}'.format(ix)] , redshift_distributions_sources['bins'][ix], ws[i])

            #IA
            IA_f = iaa.F_nla(np.mean(z_i), cosmology.Om0, rho_c1=rho_c1,A_ia = A_IA, eta=E_IA, z0=0.67,  lbar=0., l0=1e-9, beta=0.)

            # compute the galaxy density in this shell
            ngal = np.trapz(dndz_i, z_i)

            # add to mean fields using the galaxy number density as weight
            kappa_bar[ix-1,:] += ngal * kappa_i
            #gamm1_bar[ix-1,:] += ngal * gamm1_i
            #gamm2_bar[ix-1,:] += ngal * gamm2_i


            delta_bar[ix-1,:] += ngal * IA_f * delta_i



    for ix in [1,2,3,4]: 
        # compute the overall galaxy density
        ngal = np.trapz(redshift_distributions_sources['bins'][ix], redshift_distributions_sources['z'])

        # normalise mean fields by the total galaxy number density
        kappa_bar[ix-1,:] /= np.sum(ngal)
        #gamm1_bar[ix-1,:] /= np.sum(ngal)
        #gamm2_bar[ix-1,:] /= np.sum(ngal)
        delta_bar[ix-1,:] /= np.sum(ngal)

        # delta to shear and add to gamma -----
        e1,e2 = gk_inv(kappa_bar[ix-1,:],0.*kappa_bar[ix-1,:],nside,nside*2)
        gamm1_bar[ix-1,:] = e1
        gamm2_bar[ix-1,:] = e2

    Theory = theory( cosmo= Cosmo_,halofit_version='takahashi', sigma_8 = sigma_8, chistar = None)
    Theory.get_Wcmb()
    Theory.get_Wcmblog()
    Theory.limber(xtype = 'kklog',nonlinear=True) 
    cl_CMB_lensing =  Theory.clkk[0][0]


    Theory = theory( cosmo= Cosmo_,halofit_version='takahashi', sigma_8 = sigma_8, chistar = Theory.results.comoving_radial_distance(np.mean(z_i)))
    Theory.get_Wcmb()
    Theory.get_Wcmblog()
    Theory.limber(xtype = 'kklog',nonlinear=True) 
    cl_z_max =  Theory.clkk[0][0]
    
    
    import copy
    DELTA_CL_CMB = cl_CMB_lensing-cl_z_max
    DELTA_CL_CMB = np.hstack([0,DELTA_CL_CMB])
    map_ = hp.sphtfunc.synfast(DELTA_CL_CMB,nside,pixwin=True)
    cmb_lensing_map_orig  = copy.deepcopy(kappa_i) + map_


    # Add noise
    noise_uniform = np.loadtxt('/global/u2/m/mgatti/CMB_lensing_SBI/data/ACT_noise.txt')
    noise_new = hp.sphtfunc.synfast(noise_uniform[:,1], nside, pixwin=True)
    if no_noise:
        pass
    else:
        cmb_lensing_map_orig += noise_new

    #mask it 
    mask_cmb = fits.open('/global/u2/m/mgatti/CMB_lensing_SBI/data/act_mask_20220316_GAL060_rms_70.00_d2skhealpix.fits')
    mask_cmb = mask_cmb[1].data['T'].flatten()>0.9
    mask_cmb = hp.ud_grade(mask_cmb,nside_out=nside)
    cmb_lensing_map_orig[~mask_cmb] = 0.
    
    m_sources = [ params_dict['m1'], params_dict['m2'], params_dict['m3'], params_dict['m4']]#,[-0.006,-.01,-0.026,-0.032]


    for tomo_bin in [1,2,3,4]:
        m_ = 1+m_sources[tomo_bin-1]
        m_1 = 1+m_sources[tomo_bin-1]
        params_dict['dm{0}'.format(tomo_bin)]= m_1


    sources_cat = dict()
    for tomo_bin in [1,2,3,4]:

        sources_cat[tomo_bin] = dict()

        # load into memory the des y3 mock catalogue
        mcal_catalog = load_obj('/global/cfs/cdirs/des/mass_maps/Maps_final/data_catalogs_weighted_{0}'.format(tomo_bin-1))

        pix = convert_to_pix_coord(mcal_catalog['ra'], mcal_catalog['dec'], nside=nside)

        e1 = mcal_catalog['e1']
        e2 = mcal_catalog['e2']
        w  = mcal_catalog['w']


        del mcal_catalog
        gc.collect() 



        # ++++++++++++++++++++++

        n_map = np.zeros(hp.nside2npix(nside))

        unique_pix, idx, idx_rep = np.unique(pix, return_index=True, return_inverse=True)
        n_map[unique_pix] += np.bincount(idx_rep, weights=w)

        g1_ = gamm1_bar[tomo_bin-1][pix]
        g2_ = gamm2_bar[tomo_bin-1][pix]


        es1,es2 = apply_random_rotation(e1, e2)

        if no_noise:
            es1 *= 0
            es2 *= 0
        x1_sc,x2_sc = addSourceEllipticity({'shear1':g1_,'shear2':g2_},{'e1':es1,'e2':es2},es_colnames=("e1","e2"))

        g1_map = np.zeros(hp.nside2npix(nside))
        g2_map = np.zeros(hp.nside2npix(nside))



        g1_map[unique_pix] += np.bincount(idx_rep, weights=x1_sc*w)
        g2_map[unique_pix] += np.bincount(idx_rep, weights=x2_sc*w)

        mask_sims = n_map != 0.
        g1_map[mask_sims]  = g1_map[mask_sims]/(n_map[mask_sims])
        g2_map[mask_sims] =  g2_map[mask_sims]/(n_map[mask_sims])

        sources_cat[tomo_bin] = {'e1':g1_map[mask_sims],'e2':g2_map[mask_sims],'pix':np.arange(len(g2_map))[mask_sims]}


    sources_cat['params'] = params_dict


    # add CMB lensing map:

    sources_cat[5] = {'cmb':cmb_lensing_map_orig}
    return sources_cat
# Dirac prior parameter space

def draw_parameters_Dirac():
    
    Om = -1
    while ((Om<0.1) or (Om>0.5)):
        Om = np.random.normal(0.3,.075)
        
    sigma_8 = -1
    while ((sigma_8<0.5) or (sigma_8>1)):
        sigma_8 = np.random.normal(0.765,.16)
        
    A_IA = random.uniform(-2,2)
    E_IA = random.uniform(-3,3)
    
    
    w0 = -2
    while ((w0<-1.1) or (w0>-1/3.)):
        w0 = np.random.normal(-1,1/3.)
    
    h = np.random.normal(0.7022,0.0245)
    obh2 = np.random.normal(0.02237,0.00015)
    Ob = obh2/h/h
    
    
    ns = np.random.normal(0.9649,0.0063)
    mnu = np.exp(random.uniform(np.log(0.06), np.log(0.14)))



    params_dict = dict()
    params_dict['Om'] = Om
    params_dict['Ob'] = Ob
    params_dict['ns'] = ns
    params_dict['h'] = h
    params_dict['sigma_8'] = sigma_8
    params_dict['w'] = w0
    params_dict['A_IA'] = A_IA
    params_dict['E_IA'] = E_IA
    params_dict['mnu'] = mnu
    return params_dict



output_folder = '/pscratch/sd/m/mgatti/GLASS_MOPED/'

delta = dict()
delta['Om'] = 0.02
delta['sigma_8'] = 0.03
delta['w'] = 0.1
delta['A_IA'] = 0.2
delta['E_IA'] = 0.2
delta['Ob'] = 0.002
delta['ns'] = 0.04
delta['h'] = 4.0/100.



delta['m1'] = 0.05
delta['m2'] = 0.05
delta['m3'] = 0.05
delta['m4'] = 0.05
delta['dz1'] = 0.02
delta['dz2'] = 0.02
delta['dz3'] = 0.02
delta['dz4'] = 0.02

def doit(tt,mock_num):
    
    h = 0.6736
    Ob = 0.0493
    Om = 0.26
    A_IA = 0.5
    E_IA = 0
    w0 = -1.
    sigma_8 = 0.84
    ns=0.9649
    mnu=0.06
    
    params_dict = dict()
    params_dict['Om'] = Om
    params_dict['Ob'] = Ob
    params_dict['ns'] = ns
    params_dict['h'] = h
    params_dict['sigma_8'] = sigma_8
    params_dict['w'] = w0
    params_dict['A_IA'] = A_IA
    params_dict['E_IA'] = E_IA
    params_dict['mnu'] = mnu
        

    params_dict['m1'] = -0.002
    params_dict['m2'] = -0.017
    params_dict['m3'] = -0.029
    params_dict['m4'] = -0.038

    params_dict['dz1'] = 0.
    params_dict['dz2'] = 0.
    params_dict['dz3'] = 0.
    params_dict['dz4'] = 0.

    if not os.path.exists(output_folder+tt+'_p'):
        try:
            os.mkdir(output_folder+tt+'_p')
        except:
            pass
        
    if not os.path.exists(output_folder+tt+'_m'):
        try:
            os.mkdir(output_folder+tt+'_m')
        except:
            pass
        
        
    params_dict[tt] += delta[tt]/2.
    if not os.path.exists(output_folder+tt+'_p/'+str(mock_num)+'.npy'):
        sources_cat = make_mock(params_dict, fiducial_cosmo = False, no_noise=False)
        np.save(output_folder+tt+'_p/'+str(mock_num),sources_cat)

    if not os.path.exists(output_folder+tt+'_m/'+str(mock_num)+'.npy'):
        params_dict[tt] -= 2*delta[tt]/2.
        sources_cat = make_mock(params_dict, fiducial_cosmo = False, no_noise=False )
        np.save(output_folder+tt+'_m/'+str(mock_num),sources_cat)
        
        
if __name__ == '__main__':

    
   # type_ = ['Om','Ob','ns','h','sigma_8','w']

    from mpi4py import MPI 
    for type__ in delta.keys():
        #doit(type__,0)
        runs = 128
        run_count = 0
        while run_count<runs:
            comm = MPI.COMM_WORLD
            print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
            if (run_count+comm.rank)<runs:
                try:
                    doit(type__,run_count+comm.rank)
                except:
                    pass
            run_count+=comm.size
            comm.bcast(run_count,root = 0)
            comm.Barrier() 
 #

#srun --nodes=4 --tasks-per-node=32   python make_GLASS_MOPED.py 