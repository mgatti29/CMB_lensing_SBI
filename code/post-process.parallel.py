import os
from colossus.cosmology import cosmology as cosmology_colossus
from colossus.lss import mass_function
from colossus.halo import concentration
from scipy.ndimage.filters import gaussian_filter1d
import camb
import math
import timeit
import CMB_lensing_SBI
from CMB_lensing_SBI.PKDGRAV_utilities_scripts import *
from CMB_lensing_SBI.post_process_halos import *
from CMB_lensing_SBI.healpy_utils import *
from astropy.table import Table
import astropy
from astropy import units as u
from astropy.cosmology import wCDM
import numpy as np
import cosmolopy.distance as cd
import scipy
from scipy.interpolate import interp1d
import astropy.io.fits as fits
from astropy import constants as const
import frogress
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.sim_manager import UserSuppliedHaloCatalog
from mcfit import Hankel
from colossus.halo import mass_so
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq
import os
import glob
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from multiprocessing import Pool
import frogress
from CMB_lensing_SBI.PKDGRAV_utilities_scripts import *
from CMB_lensing_SBI.post_process_halos import *
from CMB_lensing_SBI.healpy_utils import *
from astropy.cosmology import wCDM
import cosmolopy.distance as cd
import scipy.interpolate

# Define the function that processes a single folder
def process_folder(folder_name):
    import os
    from colossus.cosmology import cosmology as cosmology_colossus
    from colossus.lss import mass_function
    from colossus.halo import concentration
    from scipy.ndimage.filters import gaussian_filter1d
    import camb
    import math
    import timeit
    import CMB_lensing_SBI
#    from CMB_lensing_SBI.PKDGRAV_utilities_scripts import *
#    from CMB_lensing_SBI.post_process_halos import *
#    from CMB_lensing_SBI.healpy_utils import *
    from astropy.table import Table
    import astropy
    from astropy import units as u
    from astropy.cosmology import wCDM
    import numpy as np
    import cosmolopy.distance as cd
    import scipy
    from scipy.interpolate import interp1d
    import astropy.io.fits as fits
    from astropy import constants as const
    import frogress
    from halotools.empirical_models import HodModelFactory
    from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
    from halotools.empirical_models import NFWPhaseSpace, Zheng07Sats
    from halotools.empirical_models import PrebuiltHodModelFactory
    from halotools.sim_manager import UserSuppliedHaloCatalog
    from mcfit import Hankel
    from colossus.halo import mass_so
    import pyarrow as pa
    import pandas as pd
    import pyarrow.parquet as pq
    import os
    import glob
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    import frogress
#    from CMB_lensing_SBI.PKDGRAV_utilities_scripts import *
#    from CMB_lensing_SBI.post_process_halos import *
#    from CMB_lensing_SBI.healpy_utils import *
    from astropy.cosmology import wCDM
    import cosmolopy.distance as cd
    import scipy.interpolate

    
    target_file = 'run.00100.fofstats.0'
    folder_path = os.path.join(base_dir, folder_name)
    file_path = os.path.join(folder_path, target_file)

    # Check if the file exists in the folder
    if os.path.isfile(file_path):
    
    
        folder = 'U'
        run =  folder_name
        file = folder_name+'/'

        max_redshift = 0.9


        # read cosmology 
        om,sigma8,w,ob,n_s,h,mv  = return_params('./params_run_1_Niall_U.txt'.format(folder),folder,run)

        # Se up cosmology ---------------
        cosmology = wCDM(H0= h,
                 Om0=om,
                 Ode0=1-om,
                 w0=w)

        cosmo = {'omega_M_0': om, 
         'omega_lambda_0':1-om,
         'omega_k_0':0.0, 
         'omega_b_0' : ob,
         'h':h.value/100.,
         'sigma_8' : sigma8,
         'n': n_s,
        'w':w,
        'mv':mv}

        #params = {'w0':w ,'flat': True, 'H0': h.value, 'Om0': om, 'Ob0': ob, 'sigma8':sigma8 ,'ns': n_s}
        #cosmology_colossus.addCosmology('myCosmo', params)
        #cosmo_colossus = cosmology_colossus.setCosmology('myCosmo', params,de_model = 'w0wa', w0 = w, wa = 0)
        #cosmo_colossus.w0 = w

        params = {'flat': True, 'H0': h.value, 'Om0': om, 'Ob0': ob, 'sigma8':sigma8 ,'ns': n_s}
        cosmology_colossus.addCosmology('myCosmo', params)
        cosmo_colossus = cosmology_colossus.setCosmology('myCosmo', params,de_model = 'w0wa', w0 = w, wa = 0)
        # ---------------------------------_
        z_hr = np.linspace(0,10,5001)
        d_hr = cd.comoving_distance(z_hr,**cosmo)  
        interpolated_distance_to_redshift = scipy.interpolate.CubicSpline(d_hr,z_hr)
        interpolated_redshift_to_distance = scipy.interpolate.CubicSpline(z_hr,d_hr)

        # read box size and number of particles ***********
        Lbox_Mpc = int(get_from_control_file(file + "control.par", "dBoxSize"))/(h.value/100)
        Lbox = Lbox_Mpc * (u.Mpc)
        nparts = int(get_from_control_file(file + "control.par", "nGrid"))

        # determine particle_mass ****************************
        part_mass_ = cosmology.Om0 * cosmology.critical_density(0).to(u.Msun/ u.Mpc ** 3)
        part_mass_ *= (Lbox / nparts) ** 3

        part_mass = part_mass_*150
        f_mass = Lbox**3* cosmology.critical_density(0).to(u.Msun/ u.Mpc ** 3)


        # Build reshift shell file **************************
        build_z_values_file(file,'run',out_dir=file, H0 = h.value, w = w, Lbox_Mpc = Lbox_Mpc)


        # read shell reshifts
        resume = process_resume(file + '/z_values.txt')
        resume['Lbox_Mpc'] = Lbox_Mpc
        resume['f_mass'] = f_mass
        np.save(file + 'shell_info',{'shell_info':resume,'cosmology':cosmo})
        # determine max step (for replicas)
        max_step_halocatalog = len(resume['z_far'])-int(resume['Step'][[resume['z_far']<max_redshift][0]][0])+1
        # make halo lightcone ------------------------------



        nside = 4096
        for s_ in frogress.bar(range(int(resume['Step'][-1]))):
            shell = resume['Step'][s_]
            shell = int(shell)
            if (resume['z_far'][s_]<4):
               # p =   f"{file}/run.000{shell:02d}.hpb"
                p = f"{file}/run.{int(shell):05d}.hpb" 
                if not os.path.exists(file+'/particles_{0}_{1}.parquet'.format(shell,nside)):
                    try:
                        print (p)
                        mm1 = one_healpix_map_from_basefilename(p, nside)
                        table = pa.Table.from_pandas(pd.DataFrame(mm1.astype(np.uint16)), preserve_index=False)
                        pq.write_table(table, file+'/particles_{0}_{1}.parquet'.format(shell,nside), compression='zstd')
                    except:
                        pass

        # check and delete
        import os
        import glob
        for s_ in frogress.bar(range(int(resume['Step'][-1]))):
            shell = int(resume['Step'][s_])

            if resume['z_far'][s_] < 4:
                parquet_file = f"{file}/particles_{shell}_{nside}.parquet"
                if os.path.exists(parquet_file):
                    table = pq.read_table(parquet_file)
                    # Find and remove all files matching the pattern
                    p_files = glob.glob(f"{file}/run.{shell:05d}.hpb*")
                    for p_file in p_files:
                        os.remove(p_file)
            else:
                # Find and remove all files matching the pattern
                p_files = glob.glob(f"{file}/run.{shell:05d}.hpb*")
                for p_file in p_files:
                    os.remove(p_file)


        for s_ in frogress.bar(range(int(resume['Step'][-1]))):
            shell = resume['Step'][s_]
            shell = int(shell)
            if (resume['z_far'][s_]<4):

               c__ = f'{int(shell):03}'

               p = f'{file}run.00{c__}.fofstats.0'  

               if not os.path.exists(f'{file}run.00{c__}.fofstats.parquet'):
                try:
     #          if 1==1:
                    print (c__)
                    p = f'{file}run.00{c__}.fofstats.0'
                    pkd_halo_dtype = np.dtype([("rPot", ("f4", 3)), ("minPot", "f4"), ("rcen", ("f4", 3)),
                                               ("rcom", ("f4", 3)), ("cvom", ("f4", 3)), ("angular", ("f4", 3)),
                                               ("inertia", ("f4", 6)), ("sigma", "f4"), ("rMax", "f4"),
                                               ("fMAss", "f4"), ("fEnvironDensity0", "f4"),
                                               ("fEnvironDensity1", "f4"), ("rHalf", "f4")])
                    halos = np.fromfile(p, count=-1, dtype=pkd_halo_dtype)


                    halo_center = resume['Lbox_Mpc'] * (halos["rPot"]  + halos["rcen"] + 0.5)
                    rmax = resume['Lbox_Mpc'] * halos["rMax"]
                    log10M = np.log10((halos['fMAss'] * resume['f_mass']).value)

                    num_halos = len(halos['inertia'])
                    principal_moments = np.zeros((num_halos,3))
                    principal_moments[:,0] = halos['inertia'][:,0]
                    principal_moments[:,1] = halos['inertia'][:,3]
                    principal_moments[:,2] = halos['inertia'][:,5]

                    cross = np.zeros((num_halos,3))
                    cross[:,0] = halos['inertia'][:,1]/np.sqrt(principal_moments[:,0]*principal_moments[:,1])
                    cross[:,1] = halos['inertia'][:,2]/np.sqrt(principal_moments[:,0]*principal_moments[:,2])
                    cross[:,2] = halos['inertia'][:,4]/np.sqrt(principal_moments[:,1]*principal_moments[:,2])


                    df = pd.DataFrame({
                        "halo_center": halo_center.tolist(),  # Convert arrays to list to ensure correct DataFrame format
                        "rmax":       (rmax* 1000).astype('uint16'),
                        "log10M":     (log10M* 1000).astype('uint16'),
                        "inertia_auto": list((np.log10((principal_moments)*1e20)*1000).astype('uint16')),
                        "inertia_cross":  list((10000*(1+cross)).astype('uint16')),
                        "angular": list(((np.sign(halos['angular']) *np.log10((np.abs(halos['angular'])*1e20 )+1))*1000).astype('int16'))})
                    df.to_parquet(f'{file}run.00{c__}.fofstats.parquet', engine='pyarrow', compression='brotli')
                except:
                    pass


        for s_ in frogress.bar(range(int(resume['Step'][-1]))):
            shell = int(resume['Step'][s_])
            c__ = f'{int(shell):03}'
            os.remove(file+f'run.00{c__}')
            if resume['z_far'][s_] < 4:
                parquet_file = f"{file}run.00{c__}.fofstats.parquet"
                if os.path.exists(parquet_file):
                    table = pq.read_table(parquet_file)
                    # Find and remove all files matching the pattern
                    p_files = glob.glob(f'{file}run.00{c__}.fofstats.0*')
                    for p_file in p_files:
                        os.remove(p_file)
            else:
                # Find and remove all files matching the pattern
                p_files = glob.glob(f'{file}run.00{c__}.fofstats.0*')
                for p_file in p_files:
                    os.remove(p_file)



# Main function to parallelize the folder processing
def main_parallel(base_dir, num_processes):
    # Get all folder names (e.g., run001, run002, etc.)
    folders = [f'run{i:03d}' for i in [33,44,54,88,92,399,409,418,428,433,443,452,460,469]] #range(498,513)]

    # Use multiprocessing Pool to parallelize the work
    with Pool(processes=num_processes) as pool:
        pool.map(process_folder, folders)

if __name__ == "__main__":
    # Specify the number of parallel processes
    num_processes = 20  # You can change this value
    base_dir = './'
    
    # Run the parallel processing
    main_parallel(base_dir, num_processes)
