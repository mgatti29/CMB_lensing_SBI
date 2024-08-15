import subprocess
import numpy as np
import matplotlib.pyplot as plt
from classy import Class
import numpy as np
import camb
from camb import model
from CMB_lensing_SBI.cmb_lensing_sbi_pipe import *
import healpy as hp
from pixell import enmap
from pixell import curvedsky as cs
from pixell import enmap,lensing as plensing,curvedsky, utils, enplot
from orphics import io,maps
from falafel import qe
import frogress
import re
import sys
import glob
import timeit
sys.path.append('/global/homes/m/mgatti')
import mnms
from mnms import noise_models as nm
import os
os.environ['SOFIND_SYSTEM'] = 'perlmutter'
import gc
from scipy.interpolate import interp1d






def doit(sim_num,MULTIPLE_SPLITS,add_noise,output_folder_general,cosmology):
    
    if add_noise:
        noise = 'noisy'
    else:
        noise = 'noiseless'
        
    if MULTIPLE_SPLITS:
        spl = '8split'
    else:
        spl = '1split'     
        
    outdir = output_folder_general+'/{0}_{1}_{2}_{3}/'.format(cosmology,noise,sim_num,spl)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    print ('Working in ',outdir)
    
    if not os.path.exists(outdir+'kcoadded_alms.fits'):

        ##############################################################################################################
        #
        #
        #                                             SOME CONFIGS 
        #
        #
        ##############################################################################################################


        beam_fwhm = 1.4


        # maps config --------------
        nside= 4096 # this is needed for the kappa map and the kappa_alm to lens the CMB maps. nside 1024 is good up to l ~2k., nside 4096 should pobably be our default here.
        arcmin_res_car = 1# this somehow determines also the resolution of the CAR maps. can't be too small. 
        lmax = 2500 #6000 is the default for the ACT pipeline; but we can't really do it as class can't generate lensing cls a l>2500 (unlensed ones, yes, so when we will use N-bdy sims it will be OK)


        '''

        Note: v4 ~ is the unreleased version.

        The observations were made using three
        dichroic detector modules, known as polarization arrays
        (PA), with PA4 observing in the f150 (PA4 f150) and
        f220 (PA4 f220) bands; PA5 in the f090 (PA5 f090) and
        f150 (PA5 150) bands, and PA6 in the f090 (PA6 f090)
        and f150 (PA6 f150) bands.

        the lensing map did not utilize the 220GHz data, only 90 and 150GHz, hence only pa4a (not pa4b),
        while the other arrays will have both the pa{5-6}a and pa{5-6}b data 
        https://arxiv.org/pdf/2304.05202

        '''

        path_files = '/pscratch/sd/j/jaejoonk/lensing_pipeline_data/'
        data_maps_fn_pattern = 'sim_cmb_night_%s_%s_%s_3pass_1way_set%s_map.fits'
        m = 'night'
        a_f = ['pa4_f150','pa5_f090','pa5_f150','pa6_f090','pa6_f150']
        #qids = ['pa4av4', 'pa5av4', 'pa5bv4', 'pa6av4','pa6bv4'] 
        qids = ['pa4a', 'pa5a', 'pa5b', 'pa6a','pa6b'] 
        qids = ['pa5b', 'pa6a','pa6b'] 



        array_dict = {'pa4a': 'pa4_f150', 'pa5a': 'pa5_f090', 'pa5b': 'pa5_f150','pa6a': 'pa6_f090', 'pa6b': 'pa6_f150'}


        gain_dict =  {
                     "pa4_f150": 0.9708, "pa4_f220": 1.1119, "pa5_f090": 0.9625,
                     "pa5_f150": 0.9961, "pa6_f090": 0.9660, "pa6_f150": 0.9764,
                     }


        # I assume they're the same as v4?
        pol_eff = {
                    'pa4a': 0.9584, 'pa5a': 0.9646, 'pa5b': 0.9488,
                    'pa6a': 0.9789, 'pa6b': 0.9656
                }







        global_folder = '/pscratch/sd/j/jaejoonk/lensing_pipeline_data/'
        catalog_large = global_folder + 'catalog_large/union_catalog_large_20220316.csv'
        catalog_regular = global_folder +'catalog_regular/union_catalog_regular_20220316.csv'
        nemomodel_f090 = global_folder+ '/nemomodel_f090/nemomodel_dr6_all_clustersSNR5090down2.fits'
        nemomodel_f150 = global_folder+ '/nemomodel_f150/nemomodel_dr6_all_clustersSNR5150down2.fits'
        beams_path = global_folder+ '/beams_path/20230902/'
        szbeam150 = global_folder+ '/szbeam150/s16_pa2_f150_nohwp_night_beam_tform_jitter.txt'
        szbeam90 =  global_folder+ '/szbeam90/s16_pa3_f090_nohwp_night_beam_tform_jitter.txt'
        calibration = global_folder+ 'calibration/tf_fit_dr6_%s_%s.dat'
        kcoadded_alms=  'kcoadd_data_tszsub_%s_%s.fits'


        if MULTIPLE_SPLITS:
            nsplits = 8
            data_run = False
            model_subtract_tsz = False
        else:
            nsplits = 1
            data_run = False
            model_subtract_tsz = False


        timing_container = dict()





        ##############################################################################################################
        #
        #
        #                                             GENERATE CLS FROM CAMB 
        #
        #
        ##############################################################################################################

        print ('----- GENERATE CLS FROM CAMB  ------')
        print ('')
        if not os.path.exists(outdir+'/lensed_and_beams_alms.npy'):
            if cosmology == 'fiducial':
                camb_filename = '../data/cosmo2017_10K_acc3_params.ini'

            # it's really this simple!
            pars = camb.read_ini(camb_filename)
            results = camb.get_results(pars)

            # CMB_unit set to None to exclude ~1e12 factor,
            # raw_cl to give me Cl not l(l+1)/2pi Cl
            powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)


            unlensed_cls = {'tt': powers['unlensed_scalar'][:,0], 'te': powers['unlensed_scalar'][:,3],
                            'ee': powers['unlensed_scalar'][:,1], 'bb': powers['unlensed_scalar'][:,2]}

            lensed_cls = {'tt': powers['lensed_scalar'][:,0], 'te': powers['lensed_scalar'][:,3],
                          'ee': powers['lensed_scalar'][:,1], 'bb': powers['lensed_scalar'][:,2],
                          'pp': powers['lens_potential'][:,0]}



        ##############################################################################################################
        #
        #
        #                                           MAKE ALMS for T,E,B and lens them
        #
        #
        ##############################################################################################################


        '''
        Let's generate some TT,TE.. using CLASS (it will be useful in the long run).
        I don't think it will matter for this test if these are all slightly different..

        '''

        print ('----- MAKE ALMS  ------')
        print ('')
        done_ = True


        while done_:

            try:
                if not os.path.exists(outdir+'/lensed_and_beams_alms.npy'):
                    st = timeit.default_timer()

                    # Generate TT, EE, EB, BB power spectra from unlensed alms
                    ps = np.array([[unlensed_cls['tt'], unlensed_cls['te'], 0 * unlensed_cls['te']],
                                   [unlensed_cls['te'], unlensed_cls['ee'], 0 * unlensed_cls['te']],
                                   [0 * unlensed_cls['tt'], 0 * unlensed_cls['te'], 0 * unlensed_cls['bb']]])

                    # Generate random alms (spherical harmonic coefficients) for the power spectra
                    alms_ = cs.rand_alm(ps, ainfo=None, lmax=lmax, seed=None, dtype=np.complex128, m_major=True, return_ainfo=False)

                    # Convert alms to Healpix maps
                    T_map = hp.alm2map(alms_[0], nside=nside)
                    E_map = hp.alm2map(alms_[1], nside=nside)
                    B_map = hp.alm2map(alms_[2], nside=nside)


                    np.save(outdir+'/unlensed_alms',alms_)


                    # Compute the power spectra of the maps
                    T_cl = hp.anafast(T_map)
                    E_cl = hp.anafast(E_map)
                    B_cl = hp.anafast(B_map)

                    # Assuming T_cl, E_cl, B_cl, unlensed_cls, and lmax are defined
                    ell = np.arange(len(T_cl))


                    ############################################################################################################



                    ###### compute kappa and lens maps --------------------------------------------------

                    # Make a noiseless kappa map
                    ell_ = np.arange(len(lensed_cls['pp']))  # Define ell array for the power spectrum
                    kappa_cmb = hp.synfast((lensed_cls['pp'] * (ell_ * (ell_ + 1) / 2)**2), nside=nside, lmax=lmax)  # Generate kappa map using the lensing potential power spectrum


                    # Compute alms from the kappa map


                    kappa_cmb_alm = hp.map2alm(kappa_cmb,lmax=lmax)  # Convert kappa map to spherical harmonics coefficients (alms)

                    np.save(outdir+'/kappa_cmb_alm',kappa_cmb_alm)



                    ell, emm = hp.Alm.getlm(lmax=lmax)  # Get ell and m values for the alms

                    # Convert kappa alms to phi alms (lensing potential)
                    phi_cmb_alm = kappa_cmb_alm / (ell * (ell + 1) / 2)  # Calculate the lensing potential alms
                    phi_cmb_alm[ell==0] = 1e-30
                    # Define the shape and WCS (World Coordinate System) for the map
                    shape, wcs = enmap.fullsky_geometry(res=np.deg2rad(arcmin_res_car/60.), proj="car")  # Set map resolution to 1 arcminute

                    # Generate lensed T, E, and B maps from the alms and lensing potential
                    maps_ = pixell.lensing.lens_map_curved((3, shape[0], shape[1]), wcs, phi_cmb_alm, alms_, phi_ainfo=None, maplmax=None, dtype=np.float64, spin=[0, 2], output="l", geodesic=True, verbose=False, delta_theta=None)

                    # Convert the lensed maps to alms
                    alm_TEB = pixell.curvedsky.map2alm(maps_[0], lmax=lmax, spin=[0, 2])

                    np.save(outdir+'/lensed_alms',alm_TEB)




                    # Apply beam -----------------
                    alm_TEB[0] = cs.almxfl(alm_TEB[0], lambda ell: gauss_beam(ell, beam_fwhm))
                    alm_TEB[1] = cs.almxfl(alm_TEB[1], lambda ell: gauss_beam(ell, beam_fwhm))
                    alm_TEB[2] = cs.almxfl(alm_TEB[2], lambda ell: gauss_beam(ell, beam_fwhm))

                    np.save(outdir+'/lensed_and_beams_alms',alm_TEB)

                    # Compute the power spectra from the beam-applied alms
                    cl_T_lensed_hp = hp.alm2cl(alm_TEB[0])
                    cl_E_lensed_hp = hp.alm2cl(alm_TEB[1])
                    cl_B_lensed_hp = hp.alm2cl(alm_TEB[2])

                    # Convert the beam-applied alms back to maps
                    #map_T_lensed_hp, map_E_lensed_hp, map_B_lensed_hp = hp.alm2map(alm_TEB, nside=nside, pol=False)

                    # Calculate the beam function for the given FWHM
                    tht_fwhm = np.deg2rad(beam_fwhm / 60.)
                    f_beam = np.exp(-(tht_fwhm**2) * (np.arange(lmax)**2) / (16 * np.log(2.)))





                    ############################################################################################################
                    end = timeit.default_timer()
                    done_ = False

                    del T_map
                    del E_map
                    del B_map
                    del alms_
                    del kappa_cmb
                    del kappa_cmb_alm
                    del phi_cmb_alm
                    del maps_
                    gc.collect()   
                    timing_container['generate_alms'] = end - st
                else:
                    print ('loading alms from disk')
                    alm_TEB = np.load(outdir+'/lensed_and_beams_alms.npy',allow_pickle=True)#.item()
                    done_ = False

            except:
                print ('failed, redoing it')





        ##############################################################################################################
        #
        #
        #                                           ADD NOISE
        #
        #
        ##############################################################################################################

        print ('----- ADD NOISE  ------')
        print ('')
        # Define paths to the mask and noise simulation files
        st = timeit.default_timer()

        mask_path = path_files + "/mask/act_mask_fejer1_20220316_GAL070_rms_70.00_downgrade_None.fits"
        noise_path = '/noise_sims_path/noise_sims/'

        # Read the mask map
        mask = enmap.read_map(mask_path)




        # Create an empty map to fill with noise
        full_shape, full_wcs = mask.shape, mask.wcs
        imap = enmap.empty((3,) + full_shape, full_wcs, dtype=np.float32)

        # Convert the alms to a map, convolved with the beam
        sigmap_conv = cs.alm2map(alm_TEB, imap)

        del alm_TEB
        gc.collect()

        # Apply the mask to the map (set regions with mask < 0.25 to 0)
        sigmap_conv[:, mask < 0.25] = 0

        # Handle NaN values in the map (set them to 0)
        sigmap_conv[np.isnan(sigmap_conv)] = 0




        if MULTIPLE_SPLITS:
            if add_noise:
            #if 1==1:
                # a qid is an identifier tag for a dataset, like a detector array.
                # see sofind for a list of possible qids depending on which data
                # model you load. thus, in the below, could also do ['pa4a', 'pa4b'] 
                # or ['pa6a', 'pa6b']
                #qids_ = ['pa5a', 'pa5b'] 
                #qids_ = ['pa6b']#, 'pa5b'] 
                #qids_ = np.array([q[:-2] for q in qids])
                # this will load a baseline-map noise model for act_dr6v4. could also 
                # do (for example) 'act_dr6v4_pwv_split' for pwv split maps (likewise el_split, inout_split), or `act_dr6.01` for dr6.01 products. these
                # correspond to the name of noise_model config files in the noise_model
                # product of sofind
                config_name = 'act_dr6.01_cmbmask' 

                # this will load the tiled noise model. could also do 'fdw_cmbmask'
                # for directional wavelet model (or 'tile', 'wav', or 'fdw' for
                # dr6.01; see noise_models product configs in sofind). these correspond
                # to the blocks within the config file
                noise_model_name = 'fdw'


                # if you are loading a config that requires subproduct_kwargs (e.g.,  
                # 'act_dr6v4_pwv_split' maps require a 'pwv_split' argument), you need
                # to specify which subproduct_kwargs the model will include at object
                # creation. this could be nothing (e.g., for 'act_dr6v4'),
                # {'pwv_split': ['pwv1']} (e.g, for 'act_dr6v4_pwv_split'), or may be
                # a longer list like {'inout_split': ['inout1', 'inout2']} (e.g., for
                # 'act_dr6v4_inout_split'). in the latter case, passing a pair of qids
                # will result in 4 "datasets" (the outer product of all the qids and
                # subproduct_kwargs in the list) being jointly modeled/covaried.
                subproduct_kwargs = {}
                # subproduct_kwargs = {'inout_split': ['inout1', 'inout2']}

                # instantiate NoiseModel object



                for split_i in range(nsplits):

                    for qids_ in [['pa4a', 'pa4b'] , ['pa5a', 'pa5b'], ['pa6a', 'pa6b'] ]:

                        if not os.path.exists(outdir+'/noise_{0}_{1}_{2}.npy'.format(sim_num,split_i,qids_[-1])):
                            tnm = nm.BaseNoiseModel.from_config(
                                config_name,
                                noise_model_name,
                                *qids_,
                                **subproduct_kwargs
                                )

                            # grab a sim from disk, generate on-the-fly if does not exist
                            my_sim = tnm.get_sim(split_num=split_i, sim_num=sim_num, lmax=5400)#,check_on_disk=True)#, generate=True)


                            for i,q in enumerate(qids_):
                                map_out = mnms.utils.fourier_resample(my_sim[i] ,  shape=full_shape, wcs=full_wcs)#, dtype=None)
                                np.save(outdir+'/noise_{0}_{1}_{2}'.format(sim_num,split_i,q),map_out[0])

                print ('done')
                ## grab a sim from disk, fail if does not exist on-disk
                #my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=10800, generate=False)
                #
                ## generate a sim on-the-fly whether or not exists on disk
                #my_sim = tnm.get_sim(split_num=2, sim_num=16, lmax=10800, check_on_disk=False)


        if MULTIPLE_SPLITS:

            for split in range(nsplits):
                print ('split #',split)
                for i,af in enumerate(a_f):
                    print ('--- channel ',af)
                    if not os.path.exists(f"{outdir}sim_cmb_{m}_{af}_{sim_num}_3pass_1way_set{split}_map_srcfree.fits"):

                        froot = "/global/cfs/cdirs/cmb/data/act_dr6/dr6.01/maps/" #'/home/s/sievers/kaper/scratch/maps/dr6v3_20211031/'
                        fname = "{0}/act_dr6.01_wide_{1}_night_8way_set{2}_ivar.fits".format(froot,af,split) #f"{froot}cmb_{m}_{a}_{f}_8way_coadd_ivar.fits"
                        ivar = enmap.read_map(fname)


                        #the products are the same, but in truth they are different: 
                        #the ivar for E and B are 0.5 * ivar for T. 
                        #the product is ivar for T
                        ivar_stack = []
                        ivar_stack.append(ivar)
                        ivar_stack.append(0.5*ivar)
                        ivar_stack.append(0.5*ivar)
                        ivar_stack = enmap.enmap(np.stack(ivar_stack), ivar.wcs)


                        #seed = int(i)+5
                        #wn_map = white_noise((3,)+full_shape,full_wcs,seed = seed,div=ivar)
                        if add_noise:
                            map_out = np.load(outdir+'/noise_{0}_{1}_{2}.npy'.format(sim_num,split,qids[i]),allow_pickle=True)
                            totmap = (sigmap_conv+map_out)#sims_noise[sim_num][split_i][qids[i]][0])
                        else:
                            totmap = copy.deepcopy(sigmap_conv)
                        totmap[:,mask<0.25] = 0 
                        totmap[np.isnan(totmap)] = 0
                       # totmap_dict[af] = totmap

                        #'''

                        enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{sim_num}_3pass_1way_set{split}_ivar.fits",ivar_stack)
                        enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{sim_num}_3pass_1way_set{split}_map_srcfree.fits",totmap)


                        del totmap
                        del ivar
                        del ivar_stack

                        gc.collect()


            print ('Done')

        else:


            for split in range(nsplits):
                print ('split #',split)
                for i,af in enumerate(a_f):
                    print ('--- channel ',af)
                    if not os.path.exists(f"{outdir}sim_cmb_{m}_{af}_{sim_num}_3pass_1way_set{split}_map_srcfree.fits"):
                        froot = "/pscratch/sd/j/jaejoonk/lensing_pipeline_data/ivar/" #'/home/s/sievers/kaper/scratch/maps/dr6v3_20211031/'
                        fname = "{0}/cmb_night_{1}_3pass_4way_coadd_ivar.fits".format(froot,af)

                        ivar = enmap.read_map(fname)#/0.000001


                        #the products are the same, but in truth they are different: 
                        #the ivar for E and B are 0.5 * ivar for T. 
                        #the product is ivar for T
                        ivar_stack = []
                        ivar_stack.append(ivar)
                        ivar_stack.append(0.5*ivar)
                        ivar_stack.append(0.5*ivar)
                        ivar_stack = enmap.enmap(np.stack(ivar_stack), ivar.wcs)


                        seed = int(i)+5
                        wn_map = white_noise((3,)+full_shape,full_wcs,seed = seed,div=ivar)
                        if add_noise:
                            totmap = (sigmap_conv+wn_map)
                        else:
                            totmap = copy.deepcopy(sigmap_conv)
                        totmap[:,mask<0.25] = 0 
                        totmap[np.isnan(totmap)] = 0
                       # totmap_dict[af] = totmap

                        #'''

                        enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{sim_num}_3pass_1way_set{split}_ivar.fits",ivar_stack)
                        enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{sim_num}_3pass_1way_set{split}_map_srcfree.fits",totmap)

                        del totmap
                        del wn_map
                        del ivar_stack
                        gc.collect()
            print ('Done')




        ##############################################################################################################
        #
        #
        #                                           DOWNGRADE
        #
        #
        ##############################################################################################################
        print ('----- DOWNGRADE  ------')
        print ('')
        st = timeit.default_timer()

        calibrated = True

        for qid in qids:
            print (qid)
            # get frequencies --------
            array_freq = array_dict[qid]
            array = array_freq[:3]
            freq = array_freq[4:]


            # NOISE MAPS --------------------
            if not os.path.exists(outdir+'/map_downgraded_ivar_{0}_{1}_{2}'.format(array,freq,sim_num)):
                stack = []
                for sset in np.arange(nsplits):
                    ivarfn = re.sub("_map", "_ivar", data_maps_fn_pattern) % (array, freq, sim_num,sset)
                    mul = 1. / gain_dict[array_dict[qid]] if calibrated else 1.
                    omap = enmap.read_map(outdir+ivarfn) * mul
                    stack.append(omap)
                smap = enmap.enmap(np.stack(stack), omap.wcs)

                del stack
                gc.collect()
                smap_downgraded = enmap.downgrade(smap, 2, op = np.sum)
                enmap.write_map(outdir+'/map_downgraded_ivar_{0}_{1}_{2}'.format(array,freq,sim_num), smap_downgraded)

                del smap_downgraded
                del smap
                gc.collect()

            # DATA + NOISE MAPS ------------
            if not os.path.exists(outdir+'/map_downgraded_srcfree_{0}_{1}_{2}'.format(array,freq,sim_num)):
                stack = []
                for sset in np.arange(nsplits):
                    ivarfn = re.sub("_map", "_map_srcfree", data_maps_fn_pattern) % (array, freq, sim_num,sset)
                    mul = 1. / gain_dict[array_dict[qid]] if calibrated else 1.
                    omap = enmap.read_map(outdir+ivarfn) * mul
                    stack.append(omap)
                smap = enmap.enmap(np.stack(stack), omap.wcs)

                del stack
                gc.collect()
                smap_downgraded = enmap.downgrade(smap, 2, op = np.sum)
                enmap.write_map(outdir+'/map_downgraded_srcfree_{0}_{1}_{2}'.format(array,freq,sim_num), smap_downgraded)

                del smap_downgraded
                del smap
                gc.collect()


        end = timeit.default_timer()
        timing_container['downgrade'] = end - st




        ##############################################################################################################
        #
        #
        #                                           INPAINT
        #
        #
        ##############################################################################################################

        print ('----- INPAINT  ------')
        print ('')

        st = timeit.default_timer()

        mask_path = path_files + "/mask/act_mask_fejer1_20220316_GAL070_rms_70.00_downgrade_3dg.fits"
        mask = enmap.read_map(mask_path)
        shape, wcs = mask.shape, mask.wcs

        lras,ldecs = np.loadtxt(catalog_large,unpack=True,delimiter=',')
        rras,rdecs = np.loadtxt(catalog_regular,unpack=True,delimiter=',')
        lcoords = np.asarray((ldecs,lras))
        rcoords = np.asarray((rdecs,rras))
        lrad = 10.0
        rrad = 6.0
        mask1 = maps.mask_srcs(shape,wcs,lcoords,lrad)
        mask2 = maps.mask_srcs(shape,wcs,rcoords,rrad)

        jmask = mask1 & mask2
        jmask = ~jmask




        for qid in qids:
            print (qid)
            if not os.path.exists(outdir+'/map_downgraded_srcfree_inpainted_{0}_{1}_{2}'.format(array,freq,sim_num)):
                # get frequencies --------
                array_freq = array_dict[qid]
                array = array_freq[:3]
                freq = array_freq[4:]

                # do it for the stacked maps ------------------------------------------------
                ivar_map = enmap.read_map(outdir+'/map_downgraded_ivar_{0}_{1}_{2}'.format(array,freq,sim_num))
                sig_map = enmap.read_map(outdir+'/map_downgraded_srcfree_{0}_{1}_{2}'.format(array,freq,sim_num))
                sig_map[...,mask<0.25]=0.0 ##intial maps had been masked (before downgrading) -- maybe delete here --- do we have to mask with new inpainting??
                ivar_map[...,mask<0.25]=0.0
                ip_map = gapfill_edge_conv_flat(sig_map, jmask,ivar=ivar_map) #make sure everything is getting inpainted
                ip_map[...,mask<0.25]=0.0
                enmap.write_map(outdir+'/map_downgraded_srcfree_inpainted_{0}_{1}_{2}'.format(array,freq,sim_num),ip_map)
                del ip_map
                del ivar_map
                del sig_map
                gc.collect()

        del jmask
        del mask1
        del mask2
        gc.collect()

        end = timeit.default_timer()
        timing_container['inpaint'] = end - st


        ##############################################################################################################
        #
        #
        #                                           KSPACE COADD
        #
        #
        ##############################################################################################################

        print ('----- KSPACE COADD  ------')
        print ('')
        st = timeit.default_timer()


        mask_path = path_files + "/mask/act_mask_fejer1_20220316_GAL070_rms_70.00_downgrade_3dg.fits"
        mask = enmap.read_map(mask_path)


        sz_nemo = {}
        sz90=enmap.read_map(nemomodel_f090)
        sz_nemo["f090"] = sz90
        sz150=enmap.read_map(nemomodel_f150)
        sz_nemo["f150"] = sz150


        all_maps = []
        all_ivars = []
        beam_fns = []
        specs = ['I','Q','U'] # I believe these must be TEB
        nspecs = len(specs)
        nqids = len(qids)
        nalms = get_nalms(lmax, lmax)
        noise_specs = np.zeros((nspecs, nqids, lmax+1), dtype = np.float64)
        coadded_alms_specs = np.zeros((nspecs, nqids, nalms), dtype=np.complex128)

        for q,qid in enumerate(qids):

            print (qid)
            # get frequenies
            array_freq = array_dict[qid]
            array = array_freq[:3]
            freq = array_freq[4:]


            # get beam function COADD -------------------------------
            #fn = f"{beams_path}coadd_{array}_{freq}_night_beam_tform_jitter_cmb.txt"



            # this load the stacked nsplits ----------------------
            map_splits = enmap.read_map(outdir+'/map_downgraded_srcfree_inpainted_{0}_{1}_{2}'.format(array,freq,sim_num))
            ivar_splits = enmap.read_map(outdir+'/map_downgraded_ivar_{0}_{1}_{2}'.format(array,freq,sim_num))



            # get SZ BEAM --------------------------------------------------
            this_beam = szbeam150 if freq == "f150" else szbeam90
            ls, bells = np.loadtxt(this_beam, unpack=True, usecols=[0, 1])
            bells = bells / bells[0]
            sz_beam =  interp1d(ls, bells, bounds_error=False, fill_value=0)

            # subtract the foreground map ----------------------------------
            if model_subtract_tsz:
                for split in range(nsplits):

                    fn = f"{beams_path}set{split}_{array}_{freq}_night_beam_tform_jitter_cmb.txt"
                    ls, bells = np.loadtxt(fn, unpack=True, usecols=[0, 1])
                    bells = bells / bells[0]
                    beam_q = interp1d(ls, bells, bounds_error=False, fill_value=0)

                    foreground = reconvolve_maps(sz_nemo[freq],mask,sz_beam,beam_q)
                    for j in range(len(map_splits)):
                        map_splits[j][split] = map_splits[j][split] - foreground #only subtract foreground from T map


            all_maps.append(map_splits)
            all_ivars.append(ivar_splits)


            dec_maps = [] #deconvolved beam, pixell window and kspace filter
            dec_ivars = [] #assoc ivars of decon maps (ivars are not deconvolved)


            # deconvolve beam ----------------------------------------------
            # Pixel window deconvolution, 5.4. [https://arxiv.org/pdf/2304.05202]
            for sp in frogress.bar(range(nsplits)):
                if not os.path.exists(outdir+'dmap_{0}_{1}'.format(sp,qid)):
                    if data_run:
                        fn = f"{beams_path}set{sp}_{array}_{freq}_night_beam_tform_jitter_cmb.txt"
                        ls, bells = np.loadtxt(fn, unpack=True, usecols=[0, 1])
                        bells = bells / bells[0]
                        this_beam = interp1d(ls, bells, bounds_error=False, fill_value=0)
                        smap = deconvolve_maps(map_splits[sp],mask,this_beam,lmax=6000)
                    else:
                        alm = cs.map2alm(map_splits[sp],lmax=6000)
                        alm_decon = cs.almxfl(alm,lambda ell:1/gauss_beam(ell,beam_fwhm))
                        imap = enmap.empty((3,)+mask.shape,mask.wcs,dtype=np.float32)
                        smap = cs.alm2map(alm_decon,imap)
                    dmap = kspace_mask(smap,vk_mask=[-1*90,90], hk_mask=[-1*50,50],deconvolve=True)
                    enmap.write_map(outdir+'dmap_{0}_{1}'.format(sp,qid),dmap)
                    dmap = enmap.read_map(outdir+'dmap_{0}_{1}'.format(sp,qid))


                else:
                    dmap = enmap.read_map(outdir+'dmap_{0}_{1}'.format(sp,qid))

                dec_maps.append(dmap)
                dec_ivars.append(ivar_splits[sp])


            #dec_maps = np.array(dec_maps)
            dec_ivars = np.array(dec_ivars)
            # estimate data noise --------------------------------
            bls=interp1d(np.arange(lmax),np.ones(lmax),bounds_error=False,fill_value=0)
            for ispec,spec in enumerate(specs):
                if not os.path.exists(outdir + '/1dweights_map_noise_{0}_{1}.txt'):
                    noisecl= get_datanoise(dec_maps,dec_ivars[:,:,:,:], ispec, ispec, mask,bls,beam_deconvolve=False,N=1,lmax =lmax)
                    bin_edges = np.linspace(2,len(noisecl),300).astype(int)
                    cents,cls=bandedcls(noisecl,bin_edges)
                    cls=maps.interp(cents,cls)(np.arange(len(noisecl)))
                    noise_specs[ispec, q] = cls
                    np.savetxt(outdir + '/1dweights_map_noise_{0}_{1}.txt'.format(qids[q],spec),cls)
                else:
                    noise_specs[ispec, q] = np.loadtxt(outdir + '/1dweights_map_noise_{0}_{1}.txt'.format(qids[q],spec))




            # is this the multiplicative bias from k_space cutting? ---
            ells_cal, cal = np.loadtxt(f"{calibration%(array,freq)}",unpack=True)
            cal = np.interp(np.arange(lmax),ells_cal,cal)


            # not clear here what to do

            if MULTIPLE_SPLITS:
                imap = enmap.zeros(dec_maps[0].shape,wcs=dec_maps[0].wcs)
                ivarreff = enmap.zeros(dec_maps[0].shape,wcs=dec_maps[0].wcs)
                for j in range(len(dec_ivars)):
                    imap += dec_ivars[j]*dec_maps[j]
                    ivarreff += dec_ivars[j]
                coadd_ = imap/ivarreff

            else:

                coadd_ = dec_maps[nsplits-1]


            del dec_ivars


            coadd_[~np.isfinite(coadd_)] = 0
            alms=cs.map2alm(coadd_,lmax=lmax)
            almsTcal=cs.almxfl(alms[0],1/cal)
            almsQcal=alms[1]/pol_eff[qids[q]]
            almsUcal=alms[2]/pol_eff[qids[q]]
            coadded_alms_specs[0,q]=almsTcal
            coadded_alms_specs[1,q]=almsQcal
            coadded_alms_specs[2,q]=almsUcal




        dummy_beam = np.ones(noise_specs[0].shape) # Map already beam-deconvolved
        f_shape = all_maps[0][0][0].shape
        f_wcs = all_maps[0][0][0].wcs

        kcoadd_I = kspace_coadd(coadded_alms_specs[0], dummy_beam, noise_specs[0])
        kcoadd_Q = kspace_coadd(coadded_alms_specs[1], dummy_beam, noise_specs[1])
        kcoadd_U = kspace_coadd(coadded_alms_specs[2], dummy_beam, noise_specs[2])
        kcoadd = cs.alm2map(np.array([kcoadd_I, kcoadd_Q, kcoadd_U]), enmap.empty((3,) + f_shape, f_wcs))
        #imap = enmap.empty(f_shape,f_wcs)
        #omap = cs.alm2map(kcoadd_I,imap)
        #io.plot_img(omap,down=8,filename=f"{LC.kcoadd_path}kcoadd_I.png")
        #enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/kcoadd0",enplot.plot(kcoadd[0]))

        #this doesn't work :()
        #Ealm,Balm=simgen.pureEB(kcoadd[1],kcoadd[2],mask,returnMask=0,lmax=LMAX,isHealpix=False)
        #a=cs.alm2map(np.array([kcoadd_I,Ealm,Balm]),enmap.empty((3,)+f_shape,f_wcs))
        k_alms=cs.map2alm(kcoadd,lmax=lmax).astype('complex64')
        #k_alms=k_alms.astype(np.complex128)  ##This breaks my code!!!!

        if model_subtract_tsz:
            tszsub = True
        else:
            tszsub = False
        #if args.coadd:
        #    coadd_type = "coadded"
        #else:
        coadd_type = 0 #f"{args.split}"


        outfn = outdir+'kcoadded_alms'
        hp.write_alm(outfn,k_alms,overwrite=True)

        print("DONE ------")


        end = timeit.default_timer()
        timing_container['kspace_coadd'] = end - st


        # Clean intermediate files -------
        outfn = outdir+'kcoadded_alms'
        if os.path.exists(outfn):

            files = glob.glob(outdir+'*')
            for file in files:
                if 'alms' in file:
                    print (file)
                else:
                    os.remove(file)


if __name__ == '__main__':

    '''
    #sinlge job code
    
    output_folder_general = '/pscratch/sd/m/mgatti/CMB_lensing_maps_sims/'
    MULTIPLE_SPLITS = False
    add_noise = True
    realisation_number = 0
    cosmology = 'fiducial'
    
    doit(realisation_number,MULTIPLE_SPLITS,add_noise,output_folder_general,cosmology)
    '''
    
        
    #'''
    #run in parallel
    
    output_folder_general = '/pscratch/sd/m/mgatti/CMB_lensing_maps_sims/'
    MULTIPLE_SPLITS = False
    add_noise = True
    tot_realisations = 10
    cosmology = 'fiducial'
    

    
    from mpi4py import MPI
    run_count=0
    while run_count<tot_realisations:
        #runit(tiles[run_count])
        comm = MPI.COMM_WORLD
        if (run_count+comm.rank)<tot_realisations:
            doit(run_count+comm.rank,MULTIPLE_SPLITS,add_noise,output_folder_general,cosmology)
        run_count+=comm.size
        comm.bcast(run_count,root = 0)
        comm.Barrier()
        
    #'''
    
    

#module load PrgEnv-intel
#srun --nodes=1 --tasks-per-node=8  python make_CMB_lensing_mocks_theory.py