import astropy.io.fits as fits
import healpy as hp
import numpy as np
from Moments_analysis import moments_map
import gc
import os
import glob
import copy

shifts = {}
shifts[0] = 0
shifts[1] = 0
shifts[2] = 0
shifts[3] = 0
shifts[4] = 35
shifts[5] = 35+90
shifts[6] = 90#+180
shifts[7] = 90#+180
shifts[8] = 35#+180
shifts[9] = 35+90#+180
shifts[10] = 0#+180
shifts[11] = 0#+180
shifts[12] = 0#+180
shifts[13] = 0#+180
shifts[14] = 0#+180
shifts[15] = 0#+180


def g2k_sphere(gamma1, gamma2, mask, nside=1024, lmax=2048,nosh=True):
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

    return E_map, B_map, almsE

def rotate_map(rot,delta_,pix_,nside):
    print (delta_)
    if rot ==0:
        rot_angles = [0+delta_, 0, 0]
        flip=False
        rotu = hp.rotator.Rotator(rot=rot_angles, deg=True)
        alpha, delta = hp.pix2ang(nside,pix_)
        rot_alpha, rot_delta = rotu(alpha, delta)
        if not flip:
            pix = hp.ang2pix(nside, rot_alpha, rot_delta)
        else:
            pix = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)

    if rot ==1:
        rot_angles = [-180+delta_, 0, 0]
        flip=False
        rotu = hp.rotator.Rotator(rot=rot_angles, deg=True)
        alpha, delta = hp.pix2ang(nside,pix_)
        rot_alpha, rot_delta = rotu(alpha, delta)
        if not flip:
            pix = hp.ang2pix(nside, rot_alpha, rot_delta)
        else:
            pix = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)

    if rot ==2:
        rot_angles = [-90+delta_, 0, 0]
        flip=True
        rotu = hp.rotator.Rotator(rot=rot_angles, deg=True)
        alpha, delta = hp.pix2ang(nside,pix_)
        rot_alpha, rot_delta = rotu(alpha, delta)
        if not flip:
            pix = hp.ang2pix(nside, rot_alpha, rot_delta)
        else:
            pix = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)

    if rot ==3:
        rot_angles = [-270+delta_, 0, 0]
        flip=True
        rotu = hp.rotator.Rotator(rot=rot_angles, deg=True)
        alpha, delta = hp.pix2ang(nside,pix_)
        rot_alpha, rot_delta = rotu(alpha, delta)
        if not flip:
            pix = hp.ang2pix(nside, rot_alpha, rot_delta)
        else:
            pix = hp.ang2pix(nside, np.pi-rot_alpha, rot_delta)
    return pix


def get_info(path,Dirac_sims_WL):
    folder = path.split(Dirac_sims_WL)[1].split('runs')[1].split('_')[0]
    mock_num = path.split(Dirac_sims_WL)[1].split('runs')[1].split('_')[1].split('run')[1]
    noiserel = path.split(Dirac_sims_WL)[1].split('runs')[1].split('_')[-1].split('.')[0].split('noiserel')[1]
    return folder,mock_num,int(noiserel)




def runit(runs_to_do,count):
    
    # load one map
    nside = 512



    # first : make CMB lensing map. let's add noise
    cmb_lensing_map = np.load(runs_to_do[count]['path_cmb'],allow_pickle = True).item()
    wl = np.load(runs_to_do[count]['path_WL'],allow_pickle=True).item()


    for rot in range(4):


        # add noise to the CMB map.
        # read ACT power spectrum (uniform)
        cmb_l_map = copy.deepcopy(cmb_lensing_map['CMB_lensing_map'])
        noise_uniform = np.loadtxt('../data/ACT_noise.txt')
        noise_new = hp.sphtfunc.synfast(noise_uniform[:,1], nside, pixwin=True)
        cmb_l_map += noise_new

        # rotate the map
        pix_ = rotate_map(rot,-shifts[runs_to_do[count]['noiserel']],np.arange(len(cmb_l_map)),nside)
        #cmb_l_map = cmb_l_map[pix_]


        # load mask & rotate 
        mask_cmb = fits.open('../data/act_mask_20220316_GAL060_rms_70.00_d2skhealpix.fits')
        mask_cmb = mask_cmb[1].data['T'].flatten()>0.9
        mask_cmb = hp.ud_grade(mask_cmb,nside_out=nside)
        mask_cmb = mask_cmb[pix_]

        cmb_l_map[~mask_cmb] = 0.


        # config file
        conf = dict()
        conf['j_min'] = 0
        conf['J'] = 6
        conf['B'] = 2
        conf['L'] = 2
        conf['nside'] = 512
        conf['lmax'] = conf['nside']*2
        conf['verbose'] = False
        conf['smoothing_scales'] = np.array([8.2,13.1,21.0,33.6,54.,86.,138,221.])

        
        path = output_DV+'CMB_lensing_inv_{0}_{1}_{2}_{3}.npy'.format(runs_to_do[count]['folder'],runs_to_do[count]['mock_num'],runs_to_do[count]['noiserel'],rot)
        if not os.path.exists(path):
            results = dict()
            '''
            # CMB lening only ----
            results = dict()

            conf['output_folder'] = output_intermediate+'/cmb_lensing_{0}_{1}_{2}_{3}/'.format(runs_to_do[count]['folder'],runs_to_do[count]['mock_num'],runs_to_do[count]['noiserel'],rot)

            # old moments ----------------------------------------------------
            mcal_moments = moments_map(conf)
            mcal_moments.add_map(cmb_l_map, field_label = 'k', tomo_bin = 0)
            mcal_moments.mask = mask_cmb 

            
            mcal_moments.transform_and_smooth('k_sm','k', shear = False, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = True)   
            mcal_moments.compute_moments('KK','k_sm_kE',field_label2='k_sm_kE', tomo_bins1 = [0], tomo_bins2 = [0])

            del mcal_moments.fields
            gc.collect()
            try:
                del mcal_moments.fields
                #del mcal_moments.fields_patches
                del mcal_moments.smoothed_maps
                gc.collect()
            except:
                pass
            os.system('rm -r {0}'.format(conf['output_folder']))
            results['CMB_moments'] = mcal_moments.moments
            #np.save(path,results)
            
            



            
            # WPH -----------------------
            mcal_moments = moments_map(conf)
            mcal_moments.add_map(cmb_l_map, field_label = 'k', tomo_bin = 0)
            mcal_moments.mask = mask_cmb 

            
            mcal_moments.cut_patches( nside=512, nside_small=8)
            mcal_moments.moments_pywph = dict()
            mcal_moments.moments_pywph_indexes = dict()
            mcal_moments.compute_moments_pywhm(label = 'KK',field1='k',field2='k')

            del mcal_moments.fields
            gc.collect()
            # mcal_moments.compute_moments_pywhm(label = 'NN',field1='kn',field2='kn')
            try:
                del mcal_moments.fields
                del mcal_moments.fields_patches
                gc.collect()
            except:
                pass
            os.system('rm -r {0}'.format(conf['output_folder']))

            results['CMB_WPH'] = mcal_moments.moments_pywph
            
            '''       
            #'''

            for i in range(4):

                mask_wl = np.in1d(np.arange(12*nside**2),wl[rot][i+1]['pix'])


                e1 = np.zeros(hp.nside2npix(nside))
                e2 = np.zeros(hp.nside2npix(nside))
                e1n = np.zeros(hp.nside2npix(nside))
                e2n = np.zeros(hp.nside2npix(nside))


                e1[wl[rot][i+1]['pix']] = wl[rot][i+1]['e1']
                e2[wl[rot][i+1]['pix']] = wl[rot][i+1]['e2']      
                e1n[wl[rot][i+1]['pix']] = wl[rot][i+1]['e1n']
                e2n[wl[rot][i+1]['pix']] = wl[rot][i+1]['e2n']      

                f,fb,almsE    =  g2k_sphere(e1,e2, mask_wl, nside=conf['nside'], lmax=conf['nside']*2 ,nosh=True)
                fn,fb,almsE    =  g2k_sphere(e1n,e2n, mask_wl, nside=conf['nside'], lmax=conf['nside']*2 ,nosh=True)

                f[~mask_wl] = 0
                fn[~mask_wl] = 0
                cmb_l_map[~mask_wl] = 0



                conf['output_folder'] = output_intermediate+'/cmb_lensing_{0}_{1}_{2}_{3}_{4}/'.format(runs_to_do[count]['folder'],runs_to_do[count]['mock_num'],runs_to_do[count]['noiserel'],rot,i)

                mcal_moments = moments_map(conf)
                mcal_moments.add_map(cmb_l_map, field_label = 'k', tomo_bin = 0)
                mcal_moments.add_map(f, field_label = 'wl', tomo_bin = 0)
                mcal_moments.add_map(fn, field_label = 'wln', tomo_bin = 0)
                mcal_moments.mask = mask_wl 

                ## WPH
                #mcal_moments.cut_patches( nside=512, nside_small=8)
                #mcal_moments.moments_pywph = dict()
                #mcal_moments.moments_pywph_indexes = dict()
                #mcal_moments.compute_moments_pywhm(label = 'KW',field1='k',field2='wl')

                # old moments
                mcal_moments.transform_and_smooth('k_sm','k', shear = False, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = True)   
                mcal_moments.transform_and_smooth('wl_sm','wl', shear = False, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = True)   
                mcal_moments.transform_and_smooth('wln_sm','wln', shear = False, tomo_bins = [0], overwrite = False, skip_loading_smoothed_maps = True)   
                mcal_moments.compute_moments('WK','wl_sm_kE',field_label2='k_sm_kE', tomo_bins1 = [0], tomo_bins2 = [0])
                mcal_moments.compute_moments('NK','wln_sm_kE',field_label2='k_sm_kE', tomo_bins1 = [0], tomo_bins2 = [0])

                del mcal_moments.fields
                gc.collect()
                # mcal_moments.compute_moments_pywhm(label = 'NN',field1='kn',field2='kn')
                try:
                    del mcal_moments.fields
                    #del mcal_moments.fields_patches
                    del mcal_moments.smoothed_maps
                    gc.collect()
                except:
                    pass

                #results['CMB_WL{0}_WPH'.format(i)] = mcal_moments.moments_pywph
                results['WL_CMB{0}_moments'.format(i)] = mcal_moments.moments

                os.system('rm -r {0}'.format(conf['output_folder']))

                
                
            for i in range(4):

                mask_wl = np.in1d(np.arange(12*nside**2),wl[rot][i+1]['pix'])


            
                e1 = np.zeros(hp.nside2npix(nside))
                e2 = np.zeros(hp.nside2npix(nside))
                e1n = np.zeros(hp.nside2npix(nside))
                e2n = np.zeros(hp.nside2npix(nside))


                e1[wl[rot][i+1]['pix']] = wl[rot][i+1]['e1']
                e2[wl[rot][i+1]['pix']] = wl[rot][i+1]['e2']      
                e1n[wl[rot][i+1]['pix']] = wl[rot][i+1]['e1n']
                e2n[wl[rot][i+1]['pix']] = wl[rot][i+1]['e2n']      

                f,fb,almsE    =  g2k_sphere(e1,e2, mask_wl, nside=conf['nside'], lmax=conf['nside']*2 ,nosh=True)
                fn,fb,almsE    =  g2k_sphere(e1n,e2n, mask_wl, nside=conf['nside'], lmax=conf['nside']*2 ,nosh=True)

                f[~mask_wl] = 0
                fn[~mask_wl] = 0
                cmb_l_map[~mask_wl] = 0


                conf['output_folder'] = output_intermediate+'/cmb_lensing_{0}_{1}_{2}_{3}_{4}/'.format(runs_to_do[count]['folder'],runs_to_do[count]['mock_num'],runs_to_do[count]['noiserel'],rot,i)

                mcal_moments = moments_map(conf)
                mcal_moments.add_map(cmb_l_map, field_label = 'k', tomo_bin = 0)
                mcal_moments.add_map(f, field_label = 'wl', tomo_bin = 0)
                mcal_moments.add_map(fn, field_label = 'wln', tomo_bin = 0)
                mcal_moments.mask = mask_wl 

                ## WPH
                mcal_moments.cut_patches( nside=512, nside_small=8)
                mcal_moments.moments_pywph = dict()
                mcal_moments.moments_pywph_indexes = dict()
                mcal_moments.compute_moments_pywhm(label = 'WK',field1='wl',field2='k')
                mcal_moments.compute_moments_pywhm(label = 'NK',field1='wln',field2='k')

                
                del mcal_moments.fields
                gc.collect()
                # mcal_moments.compute_moments_pywhm(label = 'NN',field1='kn',field2='kn')
                try:
                    del mcal_moments.fields
                    del mcal_moments.fields_patches
                    #del mcal_moments.smoothed_maps
                    gc.collect()
                except:
                    pass

                results['WL_CMB{0}_WPH'.format(i)] = mcal_moments.moments_pywph
                #results['CMB_WL{0}_moments'.format(i)] = mcal_moments.moments

                os.system('rm -r {0}'.format(conf['output_folder']))
                
                
            np.save(path,results)
            
            

            
if __name__ == '__main__':

    # needs to load into memory DES WL maps & CMB lensing maps.
    # then, compute second, third moments  & WPH.

    output_DV = '/global/cfs/cdirs/des/mgatti/CMB_lensing/DV/SBI_forecast/Dirac/'
    Dirac_sims_WL = '/global/cfs/cdirs/des/mgatti/Dirac_mocks/'
    output_intermediate = '/pscratch/sd/m/mgatti/PWHM/temp1/'

    # we need to load the Dirac mock, and load up the corresponding CMB lensing map
    files = glob.glob(Dirac_sims_WL+'*')
    runs_to_do = dict()
    count = 0
    for f in files:
        try:
            folder, mock_num,noiserel = get_info(f,Dirac_sims_WL)
            if (noiserel <10):
                path_cmb_lensing = Dirac_sims_WL+'runs{0}/run{1}/CMB_lensing_map_nside512.npy'.format(folder,mock_num)
                
                path = output_DV+'CMB_lensing_inv_{0}_{1}_{2}_{3}.npy'.format(folder,mock_num,noiserel,3)
                if (not os.path.exists(path)) and (os.path.exists(path_cmb_lensing)):

                    runs_to_do[count] = {'path_cmb':path_cmb_lensing,'path_WL':f, 'noiserel':noiserel,'folder':folder,'mock_num':mock_num}
                    count +=1 
        except:
            pass
        
        
        
        
    run_count=0
    #compute_phmoments(runstodo[run_count],output)
    from mpi4py import MPI 
    while run_count<len(runs_to_do.keys()):
        comm = MPI.COMM_WORLD
#
        if (run_count+comm.rank)<len(runs_to_do.keys()):
            #try:
                runit(runs_to_do,run_count+comm.rank)
            #except:
            #    pass
        #if (run_count)<len(runstodo):
        #    make_maps(runstodo[run_count])
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier() 

#srun --nodes=4 --tasks-per-node=32   python run_moments2.py