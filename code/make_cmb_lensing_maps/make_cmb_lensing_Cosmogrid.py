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




output_maps = '/global/cfs/cdirs/des/mgatti/CMB_lensing/cosmogrid_maps/'
path_sims = '/global/cfs/cdirs/des/cosmogrid/raw/fiducial/cosmo_fiducial/' 


def runit(RUN_i):

    path = output_maps+'/CMB_lensing_map_nside{0}_rel{1}.npy'.format(512,RUN_i)

    if not os.path.exists(path):
        # setup cosmology ********************
        with open(path_sims+'/run_{0}/params.yml'.format(0), "r") as f_in:
            config = yaml.safe_load(f_in.read())

        config['nside'] = 512

        h = config['H0']/100.
        Cosmo_ = cosmo(H0=config['H0'], ombh2=config['Ob']*h**2, omch2=(config['Om']-config['Ob'])*h**2,As = 2e-9,ns=config['ns'],mnu=0.02,num_massive_neutrinos=3 )


        cosmology = FlatLambdaCDM(H0= config['H0'] * u.km / u.s / u.Mpc, Om0=config['Om'])
        shell = np.load(path_sims+'/run_{0}/shells_nside=512.npz'.format(RUN_i))   


        # read shells boundaries and comoving distances ----------------
        z_bounds     = dict()                                                                                         
        z_bounds['z-high'] = np.array([shell['shell_info'][i][3] for i in range(len(shell['shell_info']))])
        z_bounds['z-low'] = np.array([shell['shell_info'][i][2] for i in range(len(shell['shell_info']))])

        i_sprt = np.argsort(z_bounds['z-low'])
        z_bounds['z-low']= (z_bounds['z-low'])[i_sprt]
        z_bounds['z-high']= (z_bounds['z-high'])[i_sprt]

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
            shell_ = shell['shells'][i_sprt[s_]]
            shell_ =  (shell_-np.mean(shell_))/np.mean(shell_)
            #shell_ = hp.ud_grade(shell_, nside_out = config['nside'])
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

    output = '/global/cfs/cdirs/des/mgatti/CMB_lensing/cosmogrid_maps'
    
    #runit(5)
    run_count = 0
    from mpi4py import MPI 
    while run_count<200:
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank)<200:
            try:
                runit(run_count+comm.rank)
            except:
                print ('failed ',run_count+comm.rank)
             #   pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 
        
#srun --nodes=4 --tasks-per-node=12   python make_cmb_lensing_Cosmogrid.py 
#