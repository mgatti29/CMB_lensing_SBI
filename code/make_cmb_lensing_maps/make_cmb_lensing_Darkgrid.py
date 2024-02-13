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






def make_maps(RUN_i):
    
    p,config = RUN_i
    output_folder = '/global/cfs/cdirs/des/mgatti/CMB_lensing/darkgrid_maps/'

    path = output_folder+'/{0}.npy'.format(p)

    if not os.path.exists(path):

        
        
        
        config['nside'] = 512
        h = config['H0']/100.
        Cosmo_ = cosmo(H0=config['H0'], ombh2=config['Ob']*h**2, omch2=(config['Om']-config['Ob'])*h**2,As = 2e-9,ns=config['ns'],mnu=0.02,num_massive_neutrinos=3 )

        cosmology = FlatLambdaCDM(H0= config['H0'] * u.km / u.s / u.Mpc, Om0=config['Om'])
        
    
        
        
        
        shell_files = glob.glob(config['folder']+'/DES-Y3-shell_*')
        z_bounds     = dict()
        z_bounds['z-high'] =np.array([float(ff.split('z-high=')[1].split('_')[0]) for ff in shell_files])
        z_bounds['z-low'] =np.array([float(ff.split('z-low=')[1].split('.fits')[0]) for ff in shell_files])
        i_sprt = np.argsort(z_bounds['z-low'])
        z_bounds['z-low']= (z_bounds['z-low'])[i_sprt]
        z_bounds['z-high']= (z_bounds['z-high'])[i_sprt]
        shell_files_sorted = np.array(shell_files)[i_sprt]
        z_bin_edges = np.hstack([z_bounds['z-low'],z_bounds['z-high'][-1]])
        
        comoving_edges = [cosmology.comoving_distance(x_) for x_ in np.array((z_bounds['z-low']))]

        z_centre = np.empty((len(comoving_edges)-1))
        for i in range(len(comoving_edges)-1):
            z_centre[i] = z_at_value(cosmology.comoving_distance,0.5*(comoving_edges[i]+comoving_edges[i+1]))

        un_ = comoving_edges[:(i+1)][0].unit
        comoving_edges = np.array([c.value for c in comoving_edges])
        comoving_edges = comoving_edges*un_

        
        
        overdensity_array = []
        for s_ in frogress.bar(range(len(z_bounds['z-high']))):
            m = fits.open(shell_files_sorted[s_])
            shell_ =m[1].data['T'].flatten()
            shell_ =  (shell_-np.mean(shell_))/np.mean(shell_)
            shell_ = hp.ud_grade(shell_, nside_out = config['nside'])
            overdensity_array.append(shell_)

        overdensity_array = np.array(overdensity_array)


        # proper ray tracing
        raytrace_object = Raytracing(overdensity_array[:], cosmology, comoving_edges[:], config['nside'], NGP = False, volume_weighted = True)
        raytrace_object.raytrace_it()



        z_max = 3.4
        try:
            imax = np.arange(len(raytrace_object.redshifts))[raytrace_object.redshifts>z_max][0]
        except:
            imax = len(raytrace_object.redshifts)-1





        Theory = theory( cosmo= Cosmo_,halofit_version='takahashi', sigma_8 = config['s8'], chistar = raytrace_object.plane_distances[imax].value)
        Theory.get_Wcmb()
        Theory.get_Wcmblog()
        Theory.limber(xtype = 'kklog',nonlinear=True) 
        cl_z_max =  Theory.clkk[0][0]


        Theory = theory( cosmo= Cosmo_,halofit_version='takahashi', sigma_8 = config['s8'], chistar = None)
        Theory.get_Wcmb()
        Theory.get_Wcmblog()
        Theory.limber(xtype = 'kklog',nonlinear=True) 
        cl_CMB_lensing =  Theory.clkk[0][0]


        DELTA_CL_CMB = cl_CMB_lensing-cl_z_max
        DELTA_CL_CMB = np.hstack([0,DELTA_CL_CMB])
        map_ = hp.sphtfunc.synfast(DELTA_CL_CMB,config['nside'],pixwin=True)
        cmb_lensing_map_orig  = copy.deepcopy(raytrace_object.convergence_raytrace[imax-1]) + map_
        cmb_lensing_map = hp.ud_grade(cmb_lensing_map_orig, nside_out = config['nside'])

        output = dict()
        output['CMB_lensing_map_{0}'.format(config['nside'])] = cmb_lensing_map_orig
        output['CMB_lensing_map'] = cmb_lensing_map
        output['camb cl'] = cl_CMB_lensing

        powers = Theory.results.get_cmb_power_spectra(Theory.pars, CMB_unit='muK')
        output['camb_powers'] = powers

        np.save(path,output)


        
        
        
        
        
        
        
        
        



        
        
if __name__ == '__main__':

    path_sims = '/global/cfs/cdirs/des/darkgrid/grid_run_1/'
    output_intermediate_maps = '/global/cfs/cdirs/des/mgatti/intermediate_darkgrid/' 
    output_folder = '/global/cfs/cdirs/des/mgatti/CMB_lensing/darkgrid_maps/'


    import glob
    runstodo=[]
    count = 0
    miss = 0

    folders_ = glob.glob(path_sims+'/cosmo_*')
    runs_cosmo = len(folders_)
    
    def get_params(name):
        #print (name)
        om = float(name.split('/')[-1].split('=')[1].split('_')[0])
        s8 = float(name.split('/')[-1].split('=')[3])  
        return om,s8

    for f in range(0,runs_cosmo):

        if not os.path.exists(output_intermediate_maps+'/meta_{0}/'.format(f)):
            try:
                os.mkdir(output_intermediate_maps+'/meta_{0}/'.format(f))
            except:
                pass

        Omegam,s8 = get_params(folders_[f])
        
        ns = 0.9649
        Ob = 0.0493
        h = 100.*0.6736
        w0 = -1.

        
        for nn in range(1):
                params_dict = dict()
                params_dict['Om'] = float(Omegam)
                params_dict['s8'] = float(s8)


                params_dict['noise'] = nn


                params_dict['folder'] = folders_[f]
                params_dict['ns'] = float(ns)
                params_dict['H0'] = float(h)
                params_dict['Ob'] = float(Ob)



                p = '{0}'.format(f)
                

                if not os.path.exists(output_folder+'/{0}.npy'.format(p)):
                    runstodo.append([p,params_dict])
                    miss+=1
                else:
                    count +=1


    #make_maps(runstodo[0])
    #'''
    run_count=0
    
    from mpi4py import MPI
    while run_count<len(runstodo):
        comm = MPI.COMM_WORLD
##
        if (run_count+comm.rank)<len(runstodo):
            try:
                make_maps(runstodo[run_count+comm.rank])
            except:
                pass
        #if (run_count)<len(runstodo):
        #    make_maps(runstodo[run_count])
        run_count+=comm.size
        comm.bcast(run_count,root = 0)

        
        


#srun --nodes=4 --tasks-per-node=12   python make_cmb_lensing_Darkgrid.py 
#