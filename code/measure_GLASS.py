import numpy as np
import glob
import os
from Moments_analysis import moments_map
import numpy as np
import gc
import pickle
import healpy as hp
import sys
import shutil

sys.path.append('/global/u2/m/mgatti/Mass_Mapping/peaks/scattering_transform')
import scattering


def g2k_sphere1(gamma1, gamma2, mask, nside=1024, lmax=2048,nosh=True):
    """
    Convert shear to convergence on a sphere. In put are all healpix maps.
    """

    gamma1_mask = gamma1 * mask
    gamma2_mask = gamma2 * mask

    KQU_masked_maps = [gamma1_mask, gamma1_mask, gamma2_mask]
    alms = hp.map2alm(KQU_masked_maps, lmax=lmax, pol=True)  # Spin transform!


    ell, emm = hp.Alm.getlm(lmax=lmax)
    if nosh:
        almsE = alms[1] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
        almsB = alms[2] * 1. * ((ell * (ell + 1.)) / ((ell + 2.) * (ell - 1))) ** 0.5
    else:
        almsE = alms[1] * 1.
        almsB = alms[2] * 1. 
    almsE[ell == 0] = 0.0
    almsB[ell == 0] = 0.0
    almsE[ell == 1] = 0.0
    almsB[ell == 1] = 0.0



    almssm = [alms[0], almsE, almsB]


    kappa_map_alm = hp.alm2map(almssm[0], nside=nside, lmax=lmax, pol=False)
    E_map = hp.alm2map(almssm[1], nside=nside, lmax=lmax, pol=False)
    B_map = hp.alm2map(almssm[2], nside=nside, lmax=lmax, pol=False)

    return E_map, B_map, almsE, almsB





def doit(xx,iii):

    
    file,p = xx
    cat = np.load(file,allow_pickle=True).item()
    
    conf = dict()
    conf['J_min'] = 5
    conf['J'] = 6
    conf['B'] = 2
    conf['L'] = 2
    conf['nside'] = 512
    conf['lmax'] = conf['nside']*2
    conf['verbose'] = False
    conf['output_folder'] = '/pscratch/sd/m/mgatti/tests/run_'+str(iii)

    if not os.path.exists(conf['output_folder']):
        os.mkdir(conf['output_folder'])
    if not os.path.exists(conf['output_folder']+'/smoothed_maps'):
        os.mkdir(conf['output_folder']+'/smoothed_maps')



    mcal_moments = moments_map(conf)
    mcal_moments.conf['smoothing_scales'] = np.array([8.2,13.1,21.0,33.6,54.,86.,138,221.])
    tomo_bins = [0,1,2,3] 
    for t in tomo_bins:

        mask_sims = np.in1d(np.arange(conf['nside']**2*12),cat[t+1]['pix'] )

        e1 = np.zeros(conf['nside']**2*12)
        e2 = np.zeros(conf['nside']**2*12)

        e1[cat[t+1]['pix']] = cat[t+1]['e1']
        e2[cat[t+1]['pix']] = cat[t+1]['e2']


        f,fb,almsE , almsB  =  g2k_sphere1(e1,e2, mask_sims, nside=conf['nside'], lmax=conf['nside']*2 ,nosh=True)

        mcal_moments.add_map(f, field_label = 'k', tomo_bin = t)
        mcal_moments.add_map(fb, field_label = 'bk', tomo_bin = t)

        if t == 3:
            mcal_moments.mask = mask_sims 
           # mcal_moments.mask = dict_temp[rel][t+1]['mask'] 

    cmb_ = cat[5]['cmb']
    cmb_[~mask_sims] = 0.
    mcal_moments.add_map(cmb_, field_label = 'k', tomo_bin = 4)


    mcal_moments.transform_and_smooth('k_sm','k', shear = False, tomo_bins = [0,1,2,3,4], overwrite = False, skip_loading_smoothed_maps = True) 
    mcal_moments.transform_and_smooth('bk_sm','bk', shear = False, tomo_bins = [0,1,2,3], overwrite = False, skip_loading_smoothed_maps = True)   



    mcal_moments.compute_moments('KK','k_sm_kE',field_label2='k_sm_kE', tomo_bins1 = [0,1,2,3,4], tomo_bins2 = [0,1,2,3,4])
    mcal_moments.compute_moments('bKK', 'bk_sm_kE',field_label2='bk_sm_kE', tomo_bins1 = [0,1,2,3], tomo_bins2 = [0,1,2,3])


    del mcal_moments.smoothed_maps
    del mcal_moments.fields
    gc.collect()


    import shutil
    shutil.rmtree(conf['output_folder'])




    bins_2 = ['0_0', '0_1', '0_2', '0_3', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3']
    bins_3 = ['0_0_0', '1_0_0', '2_0_0', '3_0_0', '0_1_1',
          '0_2_2', '0_3_3', '1_1_1', '2_1_1','2_3_3',
          '1_1_3', '1_2_2', '1_3_3', '2_2_2', '3_2_2',
          '3_3_3','0_1_2','0_1_3','0_2_3','1_2_3']
    stats = dict()
    stats['WL_2'] = []
    stats['WL_3'] = []
    stats['CMB_WL_2'] = []
    for binx in bins_2:
        stats['WL_2'].extend(mcal_moments.moments['KK'][binx])
    for binx in bins_2:
        stats['WL_2'].extend(mcal_moments.moments['bKK'][binx])
    for binx in bins_3:
        stats['WL_3'].extend(mcal_moments.moments['KK'][binx])
    for binx in ['0_4', '1_4', '2_4', '3_4']:
        stats['CMB_WL_2'].extend(mcal_moments.moments['KK'][binx])
        
        

    mcal_moments = moments_map(conf)
    mcal_moments.conf['smoothing_scales'] = np.array([8.2,13.1,21.0,33.6,54.,86.,138,221.])
    cmb_ = cat[5]['cmb']
    mcal_moments.add_map(cmb_, field_label = 'k', tomo_bin = 0)
    mcal_moments.transform_and_smooth('k_sm','k', shear = False, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = True) 
    mcal_moments.compute_moments('KK','k_sm_kE',field_label2='k_sm_kE', tomo_bins1 = [0], tomo_bins2 = [0])
    del mcal_moments.smoothed_maps
    del mcal_moments.fields
    gc.collect()


    import shutil
    shutil.rmtree(conf['output_folder'])
    stats['CMB_2'] = mcal_moments.moments['KK']['0_0']
    
    conf = dict()
    conf['j_min'] = 0
    conf['J'] = 6 #6
    conf['B'] = 2
    conf['L'] = 2
    conf['nside'] = 512
    conf['lmax'] = conf['nside']*2
    conf['verbose'] = False
    conf['output_folder'] = '/pscratch/sd/m/mgatti/tests/'+'/run_'+str(iii)

    mcal_moments = moments_map(conf)
    mcal_moments.conf['smoothing_scales'] = np.array([8.2,13.1,21.0,33.6,54.,86.,138,221.])
    tomo_bins = [0,1,2,3] 
    for t in tomo_bins:

        mask_sims = np.in1d(np.arange(conf['nside']**2*12),cat[t+1]['pix'] )

        e1 = np.zeros(conf['nside']**2*12)
        e2 = np.zeros(conf['nside']**2*12)

        e1[cat[t+1]['pix']] = cat[t+1]['e1']
        e2[cat[t+1]['pix']] = cat[t+1]['e2']


        f,fb,almsE , almsB  =  g2k_sphere1(e1,e2, mask_sims, nside=conf['nside'], lmax=conf['nside']*2 ,nosh=True)

        mcal_moments.add_map(f, field_label = 'k', tomo_bin = t)
        #mcal_moments.add_map(fb, field_label = 'bk', tomo_bin = t)

        if t == 3:
            mcal_moments.mask = mask_sims 
           # mcal_moments.mask = dict_temp[rel][t+1]['mask'] 



    mcal_moments.cut_patches( nside=512, nside_small=8)
    del mcal_moments.fields

    gc.collect()
    mcal_moments.moments_pywph = dict()
    mcal_moments.moments_pywph_indexes = dict()



    # maybe we can parallelise this ----

    print ('compute moments')
    mcal_moments.compute_moments_pywhm(label = 'KK',field1='k',field2='k')
    #mcal_moments.compute_moments_pywhm(label = 'bKK',field1='bk',field2='bk')



    try:
        #del mcal_moments.fields
        del mcal_moments.fields_patches
        #del mcal_moments.smoothed_maps
        gc.collect()
    except:
        pass



    import shutil
    shutil.rmtree(conf['output_folder'])


    
    mcal_moments.moments_pywph['KK']

    bins_2 = ['0_0', '0_1', '0_2', '0_3', '1_1', '1_2', '1_3', '2_2', '2_3', '3_3']
    bins_3 = ['1_0', '2_0', '3_0', '2_1', '3_1', '3_2']
    stats['WPH'] = []

    for binx in bins_2:
        stats['WPH'].extend(mcal_moments.moments_pywph['KK'][binx].real[:48])
    for binx in bins_3:
        stats['WPH'].extend(mcal_moments.moments_pywph['KK'][binx].real[:6])
    for binx in bins_3:
        stats['WPH'].extend(mcal_moments.moments_pywph['KK'][binx].real[12:48])

    stats['params'] = cat['params']
    np.save(output_DV+p,stats)
    

    
    
if __name__ == '__main__':

    

    import glob
    import os 
    output_DV = '/pscratch/sd/m/mgatti/GLASS_dv/LFI/'
    sims = '/pscratch/sd/m/mgatti/GLASS/'
    files = glob.glob(sims+'/*')
    files_to_run = []
    for file in files:
        p = file.split(sims)[1]
        if not os.path.exists(output_DV+p):
            files_to_run.append([file,p])

            
    from mpi4py import MPI 
    run_count = 0
    while run_count<len(files_to_run):
        comm = MPI.COMM_WORLD
        print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))
        if (run_count+comm.rank)<len(files_to_run):
            #try:
                doit(files_to_run[run_count+comm.rank],run_count+comm.rank)
            #except:
            #    pass
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 

        
#module load python
#source activate /global/common/software/des/mgatti/py38_clone
#cd /global/homes/m/mgatti/CMB_lensing_SBI/code
#srun --nodes=4 --tasks-per-node=32   python measure_GLASS.py 