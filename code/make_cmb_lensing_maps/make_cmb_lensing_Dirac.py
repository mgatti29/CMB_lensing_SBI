import CMB_lensing_SBI
from CMB_lensing_SBI.healpy_utils import *
from CMB_lensing_SBI.PKDGRAV_utilities_scripts import *
from CMB_lensing_SBI.Raytracing import *
from CMB_lensing_SBI.Theory_camb import theory,cosmo
import CMB_lensing_SBI.Bornraytrace as Bornraytrace
import os
import healpy as hp
import astropy
import astropy.io.fits as fits
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM,wCDM
from astropy.cosmology import z_at_value
import frogress
import copy
import camb
from camb import model, initialpower
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
import unyt
import yaml
from ekit import paths as path_tools
import camb
from camb import model, initialpower
import math
unyt.c.convert_to_units(unyt.km / unyt.s)
import time
import pickle

def runit(i,input_i):
    input_ = input_i[i]
    [seed,folder] = input_


    if seed <10:
        mock_number = '00{0}'.format(seed)
    elif (seed>=10) & (seed<100):
        mock_number = '0{0}'.format(seed)
    elif (seed>=100):
        mock_number = '{0}'.format(seed)

    #makes folder if it doesn't exist
    try: 
        if not os.path.exists(config['output']+'/runs{0}/'.format(folder,mock_number)):
            os.mkdir(config['output']+'/runs{0}/'.format(folder,mock_number))
    except:
        pass  

    try: 
        if not os.path.exists(config['output']+'/runs{0}/run{1}'.format(folder,mock_number)):
            os.mkdir(config['output']+'/runs{0}/run{1}'.format(folder,mock_number))
    except:
        pass  

    # path to folders
    path_folder = config['path_mocks']+'/runs{0}/'.format(folder)+'/run{1}/'.format(folder,mock_number)
    path_folder_output = config['output']+'/runs{0}//run{1}/'.format(folder,mock_number)


    # this reads the cosmological parameter of the simulations
    f = open(('/global/homes/m/mgatti/Mass_Mapping/peaks/params_run_1_Niall_{0}.txt'.format(folder)),'r')
    om_ = []
    ob_ = []
    s8_ = []
    h_ = []
    w_ = []
    ns_ = []
    mv_ = []
    for i,f_ in enumerate(f):
        if i>0:
            om_.append(float(f_.split(',')[0]))
            s8_.append(float(f_.split(',')[1]))
            w_.append(float(f_.split(',')[2]))
            ob_.append(float(f_.split(',')[3]))
            h_.append(float(f_.split(',')[4]))
            ns_.append(float(f_.split(',')[5]))
            try:
                mv_.append(float(f_.split(',')[6]))
            except:
                mv_.append(0.06)
        else:
            print (f_)

    camb_om_ = om_[seed-1]
    camb_ob_ = ob_[seed-1]
    camb_s8_ = s8_[seed-1]
    camb_h_  = h_ [seed-1]
    camb_w_  = w_ [seed-1]
    camb_ns_ = ns_[seed-1]
    camb_mv_ = mv_[seed-1]
        

    om = om_[seed-1]
    w = w_[seed-1]
    h = h_[seed-1]*100.*u.km / u.s / u.Mpc

    # read redshift information ********************************************************************

    build_z_values_file(path_folder,'run',path_folder_output)

    resume = dict()
    resume['Step'] = []
    resume['z_far'] = []
    resume['z_near'] = []
    resume['delta_z'] = []
    resume['cmd_far'] = []
    resume['cmd_near'] = []
    resume['delta_cmd'] = []

    fil_ = open(path_folder_output+'/z_values.txt')
    for z__,z_ in enumerate(fil_):

            if z__>0:
                mute = np.array(z_.split(',')).astype(float)
                resume['Step'].append(mute[0])
                resume['z_far'].append(mute[1])
                resume['z_near'].append(mute[2])
                resume['delta_z'].append(mute[3])
                resume['cmd_far'].append(mute[4]/h_[seed-1])
                resume['cmd_near'].append(mute[5]/h_[seed-1])
                resume['delta_cmd'].append(mute[6]/h_[seed-1])

    overdensity_array = []
    for s in frogress.bar(range(len(resume['Step']))):


        step_ = int(resume['Step'][-1])-s
        if step_ <10:
            zz = copy.copy('0000'+str(step_))
        elif (step_>=10) & (step_<100):
            zz =  copy.copy('000'+str(step_))
        elif (step_>=100):
            zz =  copy.copy('00'+str(step_))


        if os.path.exists(path_folder+'/run.'+zz+'.lightcone.npy'.format(zz)):
            shell_ = np.load(path_folder+'/run.'+zz+'.lightcone.npy'.format(zz),allow_pickle=True)
            shell_ =  (shell_-np.mean(shell_))/np.mean(shell_)
            shell_ = hp.ud_grade(shell_,nside_out=config['nside_intermediate'])
            overdensity_array.append(shell_)

    overdensity_array = np.array(overdensity_array)


    
    cosmology = wCDM(H0= h,
                 Om0=om,#mega_fld,
                 Ode0=1-om,#Omega_fld,
                 w0=w)

    z_near = np.array(resume['z_near'][::-1])
    z_far = np.array(resume['z_far'][::-1])
    z_bin_edges = np.hstack([z_near,z_far[-1]])
    z_bin_edges[0] = 1e-6
    comoving_edges =  cosmology.comoving_distance(z_bin_edges)

    raytrace_object = Raytracing(overdensity_array[:], cosmology, comoving_edges[:overdensity_array.shape[0]+1], config['nside_intermediate'], NGP = False, volume_weighted = True)
    
    if Born:
        kappa_lensing = np.copy(overdensity_array)*0.
        for i in frogress.bar(np.arange(2,kappa_lensing.shape[0])):
            kappa_lensing[i-2] = Bornraytrace.raytrace(cosmology.H0, cosmology.Om0,
                             overdensity_array=overdensity_array[:(i),:].T,
                             a_centre=1./(1.+raytrace_object.redshifts[:i]), 
                             comoving_edges=comoving_edges[:(i+1)])

    else:
        raytrace_object.raytrace_it()



    # CMB LENSING MAP -----------------------------------------------
    # identify the redshift of the last slice
    z_max = 3.4
    try:
        imax = np.arange(len(raytrace_object.redshifts))[raytrace_object.redshifts>z_max][0]
    except:
        imax = len(raytrace_object.redshifts)-1



    Cosmo_ = cosmo(H0=camb_h_*100., ombh2=camb_ob_*camb_h_**2, omch2=(camb_om_-camb_ob_)*camb_h_**2,As = 2e-9,ns=camb_ns_,mnu=camb_mv_,num_massive_neutrinos=3 ,w = camb_w_)
    Theory = theory( cosmo= Cosmo_,halofit_version='mead', sigma_8 = camb_s8_, chistar =None)
    
    
    # CMB lensing map up to z(imax)
    chi_cmb = Theory.results.conformal_time(0)- Theory.results.tau_maxvis
    if Born:
        kappa_cmb_lensing_imax = Bornraytrace.raytrace(cosmology.H0, cosmology.Om0,
                         overdensity_array=overdensity_array[:(imax),:].T,
                         a_centre=1./(1.+raytrace_object.redshifts[:(imax)]), 
                         comoving_edges=comoving_edges[:(imax+1)],comoving_to_CMB = chi_cmb*u.Mpc)
    else:
        pass

    

    Theory.get_Wcmb()
    Theory.get_Wcmblog()
    Theory.limber(xtype = 'kklog',nonlinear=True,zmax =raytrace_object.redshifts[imax]) 
    cl_z_max =  Theory.clkk[0][0]

    '''
    The snippet below is the same as
    powers = Theory.results.get_cmb_power_spectra(Theory.pars, CMB_unit=None, raw_cl=True)
    ell_ = np.arange(len(powers['lens_potential'][:,0]))
    powers['lens_potential'][:,0] * (ell_ * (ell_ + 1) / 2)**2
    '''
    # 
    Theory.limber(xtype = 'kklog',nonlinear=True) 
    cl_CMB_lensing =  Theory.clkk[0][0]


    DELTA_CL_CMB = cl_CMB_lensing-cl_z_max
    DELTA_CL_CMB = np.hstack([0,DELTA_CL_CMB])
    map_ = hp.sphtfunc.synfast(DELTA_CL_CMB,config['nside_intermediate'],pixwin=True)
    
    cmb_lensing_map  = copy.deepcopy(kappa_cmb_lensing_imax) + map_

    
    
    output = dict()
    output['CMB_lensing_map_{0}'.format(config['nside_intermediate'])] = cmb_lensing_map

    '''
    alms_  = hp.map2alm(cmb_lensing_map)
    lmax = hp.Alm.getlmax(len(alms_))
    ell = np.arange(lmax + 1)
    pixwin = hp.pixwin(config['nside_intermediate'])
    # Apply deconvolution to the alms
    for l in range(lmax + 1):
        factor = 1.0 / pixwin[l] if pixwin[l] != 0 else 0  # Avoid division by zero
        alms_[hp.Alm.getidx(lmax, l, np.arange(min(l, lmax - l) + 1))] *= factor
    '''
    
   # output['lensing_map_35'] = raytrace_object.convergence_raytrace[imax-1]
    output['camb cl'] = cl_CMB_lensing
    output['camb cl_35'] = cl_z_max

    
    powers = Theory.results.get_cmb_power_spectra(Theory.pars, CMB_unit=None, raw_cl=True)
    output['camb_powers'] = powers

    # This is for the SBI pipeline ~ 
    powers_muK = Theory.results.get_cmb_power_spectra(Theory.pars, CMB_unit='muK', raw_cl=True)
    output['camb_powers_muK'] = powers_muK
    output['Born'] = Born
    
    
    
    '''
    ###########################################
    Diagnostics
    ###########################################
    '''
    Diagnostics = dict()
    cl_z35 = hp.anafast(kappa_cmb_lensing_imax)
    cl_pix = hp.sphtfunc.pixwin(config['nside_intermediate'])
    Diagnostics['ratio_z3.5'] = (cl_z35[:3000])/((cl_z_max[:3000]*cl_pix[:3000]**2))

    cx = hp.anafast(cmb_lensing_map)
    Diagnostics['ratio_cmb_theory'] = cx[:3000]/(cl_CMB_lensing[:3000]*cl_pix[:3000]**2)

    
    # Let's add WL x 4 and CMBL x 4 -------------
    
    if 1 ==1:
    #try:
        # read n(z) ----------------------------
        nz_file = '//global/cfs/cdirs//des/www/y3_chains/data_vectors/2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate_6000HR.fits'     
        mu = fits.open(nz_file)

        redshift_distributions_sources = {'z':None,'bins':dict()}
        redshift_distributions_sources['z'] = mu[6].data['Z_MID']
        for ix in [1,2,3,4]:
            redshift_distributions_sources['bins'][ix] = mu[6].data['BIN{0}'.format(ix)]
            


        n_bins = 4
        # stack n(z)s in a format that our theory code will like (z,nz1,nz2,nz3,nz4)
        nzs = []
        nzs.append(mu[6].data['Z_MID'])
        for ix in [1,2,3,4]:
            nz =  mu[6].data['BIN{0}'.format(ix)]
            nz /= np.trapz(nz,mu[6].data['Z_MID'])
            # normalise and append
            nzs.append(nz)
        nzs = np.array(nzs).T
        # initialise the lensing kernel for our redshift distributions
        Theory.get_Wshear(nzs)

        # compute cls ---------------------------
        Theory.limber(xtype = 'gg',nonlinear=True) 
        Theory.limber(xtype = 'gk',nonlinear=True) 


        k_tomo = dict()
        nz_kernel_sample_dict = dict()
        cl_Born = dict()    
        dz = (z_bin_edges[1:]-z_bin_edges[:-1])
        


        for tomo_bin in [1,2,3,4]:
            k_tomo[tomo_bin] = np.zeros(hp.nside2npix(config['nside_intermediate']))
            redshift_distributions_sources['bins'][tomo_bin][250:] = 0.
            nz_sample = np.interp(raytrace_object.redshifts,redshift_distributions_sources['z'], redshift_distributions_sources['bins'][tomo_bin])
            nz_sample = nz_sample/np.sum(nz_sample*(dz[:len(nz_sample)]))
            nz_kernel_sample_dict[tomo_bin] = nz_sample*dz[:len(nz_sample)]
            for i in frogress.bar(range(len(comoving_edges)-2)):
                try:
                    k_tomo[tomo_bin]  += kappa_lensing[i]*nz_kernel_sample_dict[tomo_bin][i+1]
                except:
                    pass
        for tomo_bin in [1,2,3,4]:
            cl_Born[tomo_bin] = hp.anafast(k_tomo[tomo_bin])
            

                
        cl_Born_CMBl = dict()
        for tomo_bin in [1,2,3,4]:
            cl_Born_CMBl[tomo_bin] = hp.anafast(cmb_lensing_map,k_tomo[tomo_bin])


        WL_dict = dict()
        cl_pix = hp.sphtfunc.pixwin( config['nside_intermediate'])

        for i in range(0, 4):
            a = cl_Born[i+1][:2000]  # Adjusted to use the correct index based on your code
            b = Theory.clgg[i, i][:2000] * cl_pix[:2000]**2
            WL_dict[i] = a/b

        CMLWL_dict = dict()
        cl_pix = hp.sphtfunc.pixwin( config['nside_intermediate'])

        for i in range(0, 4):
            a = cl_Born_CMBl[i+1][:2000]  # Adjusted to use the correct index based on your code
            b = Theory.clgk[i, 0][:2000] * cl_pix[:2000]**2
            CMLWL_dict[i] = a/b    
        
        Diagnostics['WL_dict']  = WL_dict
        Diagnostics['CMLWL_dict']  = CMLWL_dict
   # except:
   #     print ('failed Diagnostics ---')

    
    
    
    output['Diagnostics'] = Diagnostics   
    
    
    
    
    '''
    ###########################################
    End Diagnostics
    ###########################################
    '''  
    
    

    path = config['output']+'/runs{0}/'.format(folder_)+'/run{1}/CMB_lensing_map_nside{2}_{3}_final.npy'.format(folder,mock_number,config['nside_intermediate'], Born_label)
    np.save(path,output)
    


if __name__ == '__main__':

    Born = True
    if Born :
        Born_label = 'Born_approx'
    else:
        Born_label = 'Raytracing'
   # folders = ['C','E']#,'I','J','K','L','M','N','O','P','Q','R','S']
    folders = ['C','E','I','J','K','L','M','N','O','P','Q','R','S']
   # folders = ['C','E','I','J','K','L','M','N','O','P','R']
    #folders = ['Q','S']
    #folders = ['C','E','I','J','K','L','M','N','R']
    #folders = ['O','P','Q','S']
    #folders = ['C','E','I','J']
    
    #folders = ['P','Q','R','S']
   # folders = ['N','O','P','Q','R','S']
   # folders = ['K','L']
#   
    folders = ['E']
   # folders = ['J','K','L','M','N','O','P','Q','R','S']
 
    runstodo=[]
    count = 0
    for folder_ in folders:
        config = dict()
        config['noise_rel'] = 0
        config['nside_intermediate'] = 2048
        config['path_mocks'] = '/global/cfs/cdirs/des/dirac_sims/original_files/'
        #/global/cfs/cdirs/des/mgatti/Dirac
        config['output'] = '/global/cfs/cdirs/des/mgatti/Dirac_mocks/' #/global/cfs/cdirs/des/dirac_sims/derived_products/'
        config['sources_bins'] = [1,2,3,4]#,2,3,4]#,2,3,4] #1,2,3,4

        #make folder:
        try:
            if not os.path.exists(config['output']+'/runs{0}/'.format(folder_)):
                os.mkdir(config['output']+'/runs{0}/'.format(folder_))
        except:
            pass

        # figure out how many realisations in the folder **************************************************
        import numpy as np
        import glob
        files = glob.glob(config['path_mocks']+'/runs{0}/'.format(folder_)+'/*')
        rel =[]
        for file in files:
            try:
                rel.append(file.split('run')[2].strip('.tar.gz') )
            except:
                pass

        #rel = [file.split('run')[2].strip('.tar.gz') for file in files]
        rel_ = []
        for r in rel:
            try:
                rel_.append(float(r))
            except:
                pass
        config['n_mocks'] = len(np.unique(rel_))
        #**************************************************************************************************
        config['nside2'] = 512



        for seed in range(config['n_mocks']+1):
            if seed <10:
                mock_number = '00{0}'.format(seed)
            elif (seed>=10) & (seed<100):
                mock_number = '0{0}'.format(seed)
            elif (seed>=100):
                mock_number = '{0}'.format(seed)
            path = config['output']+'/runs{0}/'.format(folder_)+'/run{1}/CMB_lensing_map_nside{2}_{3}_final.npy'.format(folder_,mock_number,config['nside_intermediate'], Born_label)
            if not os.path.exists(path):
                runstodo.append([seed,folder_])
            else:
                count += 1
    run_count=0

    
    #runit(5,runstodo)
    #print (count,len(runstodo))
    from mpi4py import MPI 
    while run_count<len(runstodo):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank)<len(runstodo):
            try:
                runit(run_count+comm.rank,runstodo)
            except:
                print ('failed ',runstodo[run_count+comm.rank])
             #   pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
#srun --nodes=4 --tasks-per-node=12   python make_cmb_lensing_Dirac.py 
#srun --nodes=4 --tasks-per-node=4   python make_cmb_lensing_Dirac.py 
