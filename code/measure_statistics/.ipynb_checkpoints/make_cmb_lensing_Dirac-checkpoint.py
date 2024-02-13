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

    # proper ray tracing
    raytrace_object = Raytracing(overdensity_array[:], cosmology, comoving_edges[:overdensity_array.shape[0]+1], config['nside_intermediate'], NGP = False, volume_weighted = True)
    
    
    # raytrace in 4 steps.
    '''
    final = len(raytrace_object.redshifts)-1 
    chunks = 10
    chunk_length = math.ceil(final/chunks)
    for step in range(chunks):
        path = config['output']+'/runs{0}/'.format(folder)+'/run{1}/CMB_lensing_map_nside{2}_intermediate_{3}.pkl'.format(folder,mock_number,config['nside'],step)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                raytrace_object = pickle.load(file)
        else:
            start = chunk_length*step
            end = min(chunk_length*(step+1),final)
            raytrace_object.raytrace_it(start = start, end = end)
            #np.save(path,raytrace_object, allow_pickle=True, fix_imports=False)
            with open(path, 'wb') as file:
                pickle.dump(raytrace_object, file, protocol=4)
            time.sleep(10)
        
    '''
    os.system('rm {0}'.format(config['output']+'/runs{0}/'.format(folder)+'/run{1}/*_intermediate_*'.format(folder,mock_number)))
    raytrace_object.raytrace_it()
    # CMB LENSING MAP -----------------------------------------------
    # identify the redshift of the last slice
    z_max = 3.4
    try:
        imax = np.arange(len(raytrace_object.redshifts))[raytrace_object.redshifts>z_max][0]
    except:
        imax = len(raytrace_object.redshifts)-1



    Cosmo_ = cosmo(H0=camb_h_*100., ombh2=camb_ob_*camb_h_**2, omch2=(camb_om_-camb_ob_)*camb_h_**2,As = 2e-9,ns=camb_ns_,mnu=camb_mv_,num_massive_neutrinos=3 )
    Theory = theory( cosmo= Cosmo_,halofit_version='takahashi', sigma_8 = camb_s8_, chistar = raytrace_object.plane_distances[imax].value,w = camb_w_)
    Theory.get_Wcmb()
    Theory.get_Wcmblog()
    Theory.limber(xtype = 'kklog',nonlinear=True) 
    cl_z_max =  Theory.clkk[0][0]

    Cosmo_ = cosmo(H0=camb_h_*100., ombh2=camb_ob_*camb_h_**2, omch2=(camb_om_-camb_ob_)*camb_h_**2,As = 2e-9,ns=camb_ns_,mnu=camb_mv_,num_massive_neutrinos=3 )
    Theory = theory( cosmo= Cosmo_,halofit_version='takahashi', sigma_8 = camb_s8_, chistar = None,w = camb_w_)
    Theory.get_Wcmb()
    Theory.get_Wcmblog()
    Theory.limber(xtype = 'kklog',nonlinear=True) 
    cl_CMB_lensing =  Theory.clkk[0][0]


    DELTA_CL_CMB = cl_CMB_lensing-cl_z_max
    DELTA_CL_CMB = np.hstack([0,DELTA_CL_CMB])
    map_ = hp.sphtfunc.synfast(DELTA_CL_CMB,config['nside_intermediate'],pixwin=True)
    cmb_lensing_map_orig  = copy.deepcopy(raytrace_object.convergence_raytrace[imax-1]) + map_
    cmb_lensing_map = hp.ud_grade(cmb_lensing_map_orig, nside_out = config['nside'])

    output = dict()
    output['CMB_lensing_map_{0}'.format(config['nside_intermediate'])] = cmb_lensing_map_orig
    output['CMB_lensing_map'] = cmb_lensing_map
    output['lensing_map_35'] = raytrace_object.convergence_raytrace[imax-1]
    output['camb cl'] = cl_CMB_lensing
    output['camb cl_35'] = cl_z_max

    powers = Theory.results.get_cmb_power_spectra(Theory.pars, CMB_unit='muK')
    output['camb_powers'] = powers
    
    path = config['output']+'/runs{0}/'.format(folder)+'/run{1}/CMB_lensing_map_nside{2}.npy'.format(folder,mock_number,config['nside'])
    np.save(path,output)
    


if __name__ == '__main__':

    folders = ['C','E']#,'I','J','K','L','M','N','O','P','Q','R','S']
    folders = ['C','E','I','J','K','L','M','N','O','P','Q','R','S']
    folders = ['C','E','I','J','K','L','M','N','O','P','R']
    #folders = ['Q','S']
    #folders = ['C','E','I','J','K','L','M','N','R']
    #folders = ['O','P','Q','S']
    #folders = ['C','E','I','J']
    
    #folders = ['P','Q','R','S']
   # folders = ['N','O','P','Q','R','S']
   # folders = ['K','L']
#   
    #folders = ['E']
   # folders = ['J','K','L','M','N','O','P','Q','R','S']
 
    runstodo=[]
    count = 0
    for folder_ in folders:
        config = dict()
        config['noise_rel'] = 0
        config['nside_intermediate'] = 1024
        config['nside'] = 512
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
            path = config['output']+'/runs{0}/'.format(folder_)+'/run{1}/CMB_lensing_map_nside{2}.npy'.format(folder_,mock_number,config['nside'])
            if not os.path.exists(path):
                runstodo.append([seed,folder_])
            else:
                count += 1
    run_count=0

    
    #runit(5,runstodo)
    print (count,len(runstodo))
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
#srun --nodes=4 --tasks-per-node=8   python make_cmb_lensing_Dirac.py 
