import subprocess
import numpy as np
import matplotlib.pyplot as plt
from classy import Class
import numpy as np
import camb
from camb import model
import CMB_lensing_SBI
from CMB_lensing_SBI.cmb_lensing_sbi_pipe import *
import healpy as hp
from pixell import enmap
from pixell import curvedsky as cs
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
from orphics import io,maps
from falafel import qe
import frogress
import copy
import re
import glob
import sys
import timeit
import dill as pickle  # Use dill instead of pickle
from tqdm import tqdm
from pixell.mpi import FakeCommunicator
from mpi4py import MPI
sys.path.append('/global/homes/m/mgatti')
import mnms
from mnms import noise_models as nm
import os
os.environ['SOFIND_SYSTEM'] = 'perlmutter'
import gc
from scipy.interpolate import interp1d
from orphics import maps,io,stats
from orphics import cosmology as cosmology_orphics
import sys
from falafel import qe
sys.path.append('/global/homes/m/mgatti/Mass_Mapping/CMB_lensing/extra/tempura/')
import pytempura
import frogress
import multiprocessing as mp
from mpi4py import MPI
from astropy.coordinates import Angle
from astropy import units as u

def calibrate():
    

    ##############################################################################
    #    
    #    
    #                              FILTER
    #
    #    
    ##############################################################################

    '''
    Everything below is just for calibration, There's no "target" sim or data.
    This code computes a number of corrections based on noisy simulated Data:

    R_src_tt,Nl_g_bh,Als,Nl_g,Nl_c

    These corrections will be used later to correct the power spectrum.
    '''
    mask_path = path_files + "/mask/act_mask_fejer1_20220316_GAL070_rms_70.00_downgrade_3dg.fits"
    mask = enmap.read_map(mask_path)

    if SIMPLE_SIM:
        arcmin_res_car = 1
        dec_cut = Angle(np.asarray((-89, 89)), unit=u.degree).rad
        shape, wcs = enmap.band_geometry(dec_cut,res=np.deg2rad(arcmin_res_car/60.),proj='car')

        full_shape = copy.deepcopy(shape)
        full_wcs = copy.deepcopy(wcs)
    else:    
        full_shape, full_wcs = mask.shape, mask.wcs

    # Load many noisy maps into memory ---
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        print ('----- FILTER  ------')
        print ('')
        print ('nsims_noisy ',nsims_noisy)

    noisy_sims_alms = []
    count = 0
    for i in frogress.bar(range(nsims_noisy)):
       # try:        
        if SIMPLE_SIM:
            outdir = output_folder_general+'/SIMPLE_{0}_{1}_{2}_{3}/'.format('fiducial','noisy',i,spl)
            cls = np.load(outdir+'TEB_smoothed_cls.npy',allow_pickle=True)
            if cls.shape[1] ==mlmax +1 :
                noisy_sims_alms.append(cls) 
                count +=1 
        else:
            outdir = output_folder_general+'/{0}_{1}_{2}_{3}/'.format('fiducial','noisy',i,spl)


            cls = np.load(outdir+'TEB_smoothed_cls.npy',allow_pickle=True)
            if cls.shape[1] ==mlmax +1 :
                noisy_sims_alms.append(cls) 
                count +=1      
       # except:
       #     pass

    print ('')
    print ('Total number of noisy sims ',count)


    noisy_sims_alms=np.mean(noisy_sims_alms,axis=0)
    tcls={}
    tcls['TT'] = noisy_sims_alms[0][:mlmax+1]
    tcls['TE'] = noisy_sims_alms[-1][:mlmax+1]
    tcls['EE'] = noisy_sims_alms[1][:mlmax+1]
    tcls['BB'] = noisy_sims_alms[2][:mlmax+1]


    '''
    LOAD THEORY CLS at fiducial cosmology -------------------------------------------------
    '''
    thloc = '../data/cosmo2017_10K_acc3'
    ls = np.arange(mlmax+1)
    ucls = {}


    grad = True
    theory = cosmology_orphics.loadTheorySpectraFromCAMB(thloc,get_dimensionless=False)
    ells,gt,ge,gb,gte = np.loadtxt(f"{thloc}_camb_1.0.12_grads.dat",unpack=True,usecols=[0,1,2,3,4])
    ucls['TT'] = maps.interp(ells,gt)(ls) if grad else theory.lCl('TT',ls)
    ucls['TE'] = maps.interp(ells,gte)(ls) if grad else theory.lCl('TE',ls)
    ucls['EE'] = maps.interp(ells,ge)(ls) if grad else theory.lCl('EE',ls)
    ucls['BB'] = maps.interp(ells,gb)(ls) if grad else theory.lCl('BB',ls)
    ucls['kk'] = theory.gCl('kk',ls)


    est_list = ['TT', 'TE', 'TB', 'EB', 'EE', 'MV', 'MVPOL']
    est = 'MV'

    # Do point source-hardening ----------------------------------


    e2='src'
    est_list.append(e2)
    est_list.append("TT") # WHY ADD TT AGAIN?? ASK FRANK

    # profile hardening ------------------
    profile=np.loadtxt(path_files+'profile/tsz_profile5000.txt')
    Als = pytempura.get_norms(est_list,ucls,ucls,tcls,lmin,lmax,k_ellmax=mlmax,profile=profile)
    R_src_tt = pytempura.get_cross(e2,'TT',ucls,tcls,lmin,lmax,k_ellmax=mlmax,profile=profile)



    ls = np.arange(Als[est][0].size)
    # Convert to noise per mode on lensing convergence ?? ASK FRANK ABOUT THIS
    e1 =est.upper()
    Nl_g = Als[e1][0] * (ls*(ls+1.)/2.)**2.
    Nl_c = Als[e1][1] * (ls*(ls+1.)/2.)**2.

    def bias_hardened_n0(Nl,Nlbias,Cross):
        ret = Nl*0
        ret[1:] = Nl[1:] / (1.-Nl[1:]*Nlbias[1:]*Cross[1:]**2.)
        return ret

    Nl_g_bh = bias_hardened_n0(Als[e1][0],Als[e2],R_src_tt) * (ls*(ls+1.)/2.)**2.

    if rank == 0:
        np.savetxt(base+'R_src_tt_{0}'.format(lmax),R_src_tt)
        np.savetxt(base+'Nl_g_bh_{0}'.format(lmax),Nl_g_bh)
        np.save(base+'Als_lmin_{0}'.format(lmax),Als)
        np.savetxt(base+'N0g_lmin_{0}'.format(lmax),Nl_g)
        np.savetxt(base+'N0c_lmin_{0}'.format(lmax),Nl_c)
        np.save(base+'tcls',tcls)
        np.save(base+'ucls',ucls)
        print ('Done, saved calibration factors')

    comm.Barrier()


    ##############################################################################
    #    
    #    
    #                              MEAN FIELD
    #
    #    
    ##############################################################################


    print ('----- MEAN FIELD  ------')
    print ('')



    '''
    Everything below is just for calibration, There's no "target" sim or data.
    It reads a number of noiseless simulations, splitted in two batches, and it computes the mean field correction.
    '''
    from pixell import curvedsky as cs

    def process_split(i, spl, output_folder_general, nsplits,noisy_sims_alms, lmin, lmax, donoisy = False, dopair = False):


        # Initialize pixelization and qfunc inside the process
        px = qe.pixelization(shape=full_shape, wcs=full_wcs, nside=None)
        qfunc = get_qfunc(px, ucls, mlmax, e1, Al1=Als[e1], est2='SRC', Al2=Als['src'], Al3=Als['TT'], R12=R_src_tt, profile=profile)

        if nsplits == 4:
            phi_names=['phi_xy_X','phi_xy01','phi_xy02','phi_xy03','phi_xy12','phi_xy13','phi_xy23','phi_xy_x0','phi_xy_x1','phi_xy_x2','phi_xy_x3']
        else:
            phi_names=['phi_xy00']

        count = 0 
        cltt = noisy_sims_alms[0]
        clee = noisy_sims_alms[1]
        clbb = noisy_sims_alms[2]
        clte = noisy_sims_alms[3]

        Xdat = {}

        # Determine the simulation type
        sim_type = 'SIMPLE_fiducial' if SIMPLE_SIM else 'fiducial'

        # Determine the noise type
        noise_type = 'noisy' if donoisy else 'noiseless'

        # Determine if it's a pair or not
        pair_suffix = '_pair' if dopair else ''

        # Build the output directory
        outdir = f'{output_folder_general}/{sim_type}_{noise_type}_{i}_{spl}{pair_suffix}/'



        if not os.path.exists(outdir + '/xy.npy'):
            for split in range(nsplits):

                # load maps ------
                fname = outdir + 'kcoadded_alms_reshaped'
                alms = hp.read_alm(fname, hdu=(1, 2, 3))

                ls = np.arange(len(cltt))
                nells_T = maps.interp(ls, cltt) 
                nells_E = maps.interp(ls, clee)
                nells_B = maps.interp(ls, clbb)
                filt_t = 1. / (nells_T(ls))  # use as inverse filter the mean of the noisy simulated maps.
                filt_e = 1. / (nells_E(ls))  # use as inverse filter the mean of the noisy simulated maps.
                filt_b = 1. / (nells_B(ls))  # use as inverse filter the mean of the noisy simulated maps.
                talm = qe.filter_alms(alms[0].copy(), filt_t, lmin=lmin, lmax=lmax)
                ealm = qe.filter_alms(alms[1].copy(), filt_e, lmin=lmin, lmax=lmax)
                balm = qe.filter_alms(alms[2].copy(), filt_b, lmin=lmin, lmax=lmax)
                Xdat[split] = np.array([talm, ealm, balm])

            if nsplits == 4:
                xy = four_split_phi(Xdat[0], Xdat[1], Xdat[2], Xdat[3], q_func1=qfunc)
            elif nsplits == 1:
                xy = np.array([plensing.phi_to_kappa(qfunc(Xdat[0], Xdat[0]))])

            # Save xy.npy
            np.save(outdir + '/Xdat.npy', Xdat)
            np.save(outdir + '/xy.npy', xy)



    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start with run_count = 0, but each process handles tasks based on rank
    run_count = rank

    # Loop over tasks
    while run_count < nsims_noiseless:
        # Each process works on its own task
        if 1==1:
      #  try:
            process_split(run_count,spl, output_folder_general, nsplits, noisy_sims_alms, lmin, lmax, donoisy = False)
            process_split(run_count,spl, output_folder_general, nsplits, noisy_sims_alms, lmin, lmax, donoisy = True)
            process_split(run_count,spl, output_folder_general, nsplits, noisy_sims_alms, lmin, lmax, donoisy = True, dopair=True)
      
       # except:
       #     pass
        # Increment run_count by the size of the communicator to move to the next task for this process
        run_count += size
        
    comm.Barrier()

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:


        if nsplits == 4:
            phi_names=['phi_xy_X','phi_xy01','phi_xy02','phi_xy03','phi_xy12','phi_xy13','phi_xy23','phi_xy_x0','phi_xy_x1','phi_xy_x2','phi_xy_x3']
        else:
            phi_names=['phi_xy00']

        for stack in range(2):

            s = dict()
            for i in range(len(phi_names)):
                s['r'+phi_names[i]+'f'] = []
                s['i'+phi_names[i]+'f'] = []
                s['r'+phi_names[i]+'fc'] = []
                s['i'+phi_names[i]+'fc'] = []
            if stack == 0:
                init = 0
                end = int(nsims_noiseless/2)       
            else:
                init = int(nsims_noiseless/2)
                end = nsims_noiseless
            print ('')
            print ('Doing stack #{0} [sims {1} -{2}]'.format(stack,init,end))

            for i in frogress.bar(range(init,end)):
                if SIMPLE_SIM:
                    outdir = f'{output_folder_general}/SIMPLE_fiducial_noiseless_{i}_{spl}/'
                else:
                    outdir = f'{output_folder_general}/fiducial_noiseless_{i}_{spl}/'
                xy = np.load(outdir+'/xy.npy',allow_pickle=True)



                xy_a=[]
                xy_c=[]
                for i in range(len(xy)):
                    xy_a.append(xy[i][0])  #DON'T REALLY UNDERSTAND THIS -- ASK FRANK
                    xy_c.append(xy[i][1])
                xy_a=np.array(xy_a)
                xy_c=np.array(xy_c)

                for i in range(len(phi_names)):
                    s['r'+phi_names[i]+'f'] .append(xy_a[i].real)
                    s['i'+phi_names[i]+'f'] .append(xy_a[i].imag)
                    s['r'+phi_names[i]+'fc'] .append(xy_c[i].real)
                    s['i'+phi_names[i]+'fc'] .append(xy_c[i].imag)


            for i in range(len(phi_names)):
                mfalm= np.mean(np.array(s['r'+phi_names[i]+'f']),axis=0) + 1j*np.mean(np.array(s['i'+phi_names[i]+'f']),axis=0) 
                mfalmc=np.mean(np.array(s['r'+phi_names[i]+'fc']),axis=0) + 1j*np.mean(np.array(s['i'+phi_names[i]+'fc']),axis=0) 
                hp.write_alm(base+'mf_grad_fn_{0}_{1}'.format(phi_names[i], stack),mfalm,overwrite=True) 
                hp.write_alm(base+'mf_curl_fn_{0}_{1}'.format(phi_names[i], stack),mfalmc,overwrite=True)



        
        
    print ('Done mean field')


    comm.Barrier()





    ##############################################################################
    #    
    #    
    #                              MC BIAS
    #
    #    
    ##############################################################################
 
    
    print ('----- N1 BIAS  ------')
    print ('')


    px = qe.pixelization(shape=full_shape, wcs=full_wcs, nside=None)
    qfunc = get_qfunc(px, ucls, mlmax, e1, Al1=Als[e1], est2='SRC', Al2=Als['src'], Al3=Als['TT'], R12=R_src_tt, profile=profile)

    def process_task(i):
        if i>2:
            if not os.path.exists(base+'/n1evals_term1_{0}.npy'.format(i)):
                powfunc = lambda x,y: cs.alm2cl(x,y)

                
                def get_kmap(nsim, pair=False):
                    # Determine the simulation type
                    sim_type = 'SIMPLE_fiducial' if SIMPLE_SIM else 'fiducial'

                    # Determine if it's a pair or not
                    pair_suffix = '_pair' if pair else ''

                    # Build the output directory
                    outdir = f'{output_folder_general}/{sim_type}_noisy_{nsim}_{spl}{pair_suffix}/'

                    # Load Xdat
                    Xdat = np.load(outdir + '/Xdat.npy', allow_pickle=True).item()[0]

                    return Xdat


                Xs = get_kmap(i + 1)  # S
                Ysp = get_kmap(i + 2)  # S'
                Xsk = get_kmap(i)  # Sphi
                Yskp = get_kmap(i, pair=True)  # Sphi'
                
                
                
                
                qa_Xsk_Yskp = 0.5 * (qfunc(Xsk, Yskp) + qfunc(Yskp, Xsk))  # (Sphi,Sphi'), (Sphi',Sphi)
                qa_Xs_Ysp = 0.5 * (qfunc(Xs, Ysp) + qfunc(Ysp, Xs))  # (sim_i+1,Sphi'), (Sphi',sim_i)
                term = 2 * (powfunc(qa_Xsk_Yskp, qa_Xsk_Yskp) - powfunc(qa_Xs_Ysp, qa_Xs_Ysp))
                
                
                np.save(base+'/n1evals_term_{0}.npy'.format(i),term)


  
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start with run_count = 0, but each process handles tasks based on rank
    run_count = rank 

    # Loop over tasks
    while run_count < nsims_noisy-2:

        process_task(run_count)

        run_count += size
        
    comm.Barrier()


if __name__ == '__main__':


    '''
    This script runs 
    filter.py
    mean_field.py
    mcn1.py
    
    These scripts don't need a "target" simulations, but only noisy/noiseless sims at fiducial cosmology.
    '''
    
    output_folder_general = '/pscratch/sd/m/mgatti/CMB_lensing_maps_sims_1split_masked/'
    
    path_files = '/pscratch/sd/j/jaejoonk/lensing_pipeline_data/'

    MULTIPLE_SPLITS = False
    SIMPLE_SIM = False
    lmin = 600
    lmax = 2500
    mlmax =  4000
    nsims_noisy = 40
    nsims_noiseless = 40 
    nsimsB = 40
    target_sim_i = 0

    if MULTIPLE_SPLITS:
        spl = '8split'
        nsplits = 8
    else:
        spl = '1split'     
        nsplits = 1


    if SIMPLE_SIM:
        spl = '1split'
        nsplits = 1

        
    if not SIMPLE_SIM:
        outdir_target = f'{output_folder_general}/fiducial_noisy_{target_sim_i}_{spl}/'
        outdir_target_noiseless = f'{output_folder_general}/fiducial_noiseless_{target_sim_i}_{spl}/'
    else:
        outdir_target = f'{output_folder_general}/SIMPLE_fiducial_noisy_{target_sim_i}_{spl}/'
        outdir_target_noiseless = f'{output_folder_general}/SIMPLE_fiducial_noiseless_{target_sim_i}_{spl}/'

        
    # This folder will contain the calibration quantities[computed at fiducial cosmology] for our maps.
    if SIMPLE_SIM:
        base = output_folder_general+'/SIMPLE_Calibration_quantities_{0}_{1}/'.format('fiducial',spl)
        if not os.path.exists(base):
            try:
                os.mkdir(base)     
            except:
                pass
    else:
        base = output_folder_general+'/Calibration_quantities_{0}_{1}/'.format('fiducial',spl)
        if not os.path.exists(base):
            try:
                os.mkdir(base)
            except:
                pass
        
    # Calibration -----
    # this computes calibration factors at fiducial cosmology -----------------------------------------------
    calibrate()
    
    #'''
    
    ##############################################################################
    #    
    #    
    #                              RDN0 terms -----
    #
    #    
    ##############################################################################


    
    mask_path = path_files + "/mask/act_mask_fejer1_20220316_GAL070_rms_70.00_downgrade_3dg.fits"
    mask = enmap.read_map(mask_path)
    
    ucls = np.load(base+'ucls.npy',allow_pickle=True).item()

    if SIMPLE_SIM:
        arcmin_res_car = 1
        dec_cut = Angle(np.asarray((-89, 89)), unit=u.degree).rad
        shape, wcs = enmap.band_geometry(dec_cut,res=np.deg2rad(arcmin_res_car/60.),proj='car')

        full_shape = copy.deepcopy(shape)
        full_wcs = copy.deepcopy(wcs)
    else:    
        full_shape, full_wcs = mask.shape, mask.wcs

        
        
    est_list = ['TT', 'TE', 'TB', 'EB', 'EE', 'MV', 'MVPOL']
    est = 'MV'
    e2='src'
    est_list.append(e2)
    est_list.append("TT") # WHY ADD TT AGAIN?? ASK FRANK
    e1 =est.upper()
    Als = np.load(base+'Als_lmin_{0}.npy'.format(lmax),allow_pickle=True).item()
    R_src_tt = np.loadtxt(base+'R_src_tt_{0}'.format(lmax))
        
    profile=np.loadtxt(path_files+'profile/tsz_profile5000.txt')
        
    
    
    
    px = qe.pixelization(shape=full_shape, wcs=full_wcs, nside=None)
    qfunc = get_qfunc(px, ucls, mlmax, e1, Al1=Als[e1], est2='SRC', Al2=Als['src'], Al3=Als['TT'], R12=R_src_tt, profile=profile)
    #qfunc = get_qfunc(px, ucls, mlmax, e1, Al1=Als[e1])#, est2='SRC', Al2=Als['src'], Al3=Als['TT'], R12=R_src_tt, profile=profile)

    # RDN0 corrections ---
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Start with run_count = 0, but each process handles tasks based on rank
    run_count = rank

    Xdat = np.load(outdir_target + '/Xdat.npy', allow_pickle=True).item()
    # Loop over tasks
    while run_count < nsims_noisy-1:
        if (run_count != target_sim_i):
            # Each process works on its own task
            if SIMPLE_SIM:
                outdir0 = output_folder_general+'/SIMPLE_{0}_{1}_{2}_{3}/'.format('fiducial','noisy',run_count,spl)
                outdir1 = output_folder_general+'/SIMPLE_{0}_{1}_{2}_{3}/'.format('fiducial','noisy',run_count+1,spl)  

            else:
                outdir0 = output_folder_general+'/{0}_{1}_{2}_{3}/'.format('fiducial','noisy',run_count,spl)   
                outdir1 = output_folder_general+'/{0}_{1}_{2}_{3}/'.format('fiducial','noisy',run_count+1,spl)  

            if not os.path.exists(outdir_target+'qfunc_YX_{0}.npy'.format(run_count+1)):
                Xs  = np.load(outdir0 + '/Xdat.npy', allow_pickle=True).item()
                Xs1 = np.load(outdir1 + '/Xdat.npy', allow_pickle=True).item()

                np.save(outdir_target+'qfunc_XA_{0}'.format(run_count+1),qfunc(Xdat[0], Xs[0]))
                np.save(outdir_target+'qfunc_AX_{0}'.format(run_count+1),qfunc(Xs[0], Xdat[0]))       

                np.save(outdir_target+'qfunc_XY_{0}'.format(run_count+1),qfunc(Xs[0],Xs1[0]))
                np.save(outdir_target+'qfunc_YX_{0}'.format(run_count+1),qfunc(Xs1[0], Xs[0]))      
                # Increment run_count by the size of the communicator to move to the next task for this process
            run_count += size
    comm.Barrier()
    #'''

'''
module load python
source activate cmb_lensing_env
module load PrgEnv-intel
srun --nodes=4 --tasks-per-node=3 python calibration.py
'''
