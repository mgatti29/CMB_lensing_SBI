import numpy as np
import os
from astropy.io import fits
from astropy.cosmology import z_at_value
from astropy.table import Table
from astropy import units as u
import frogress
import gc
import math
import healpy as hp

def _far_point_box(box_size, x_i, y_i, z_i):
    """
    Calculate the distance to the farthest point of a box from the origin,
    accounting for the box size and coordinates of the replica.

    Parameters:
    box_size (float): The size of the box.
    x_i (int): The x-coordinate of the replica.
    y_i (int): The y-coordinate of the replica.
    z_i (int): The z-coordinate of the replica.

    Returns:
    float: The distance to the farthest point of the box.
    """
    
    # Calculate the coordinates of the farthest point of the box
    x_b = (box_size * x_i)
    y_b = (box_size * y_i)
    z_b = (box_size * z_i)
    
    # Calculate the distance from the origin to the farthest point
    d_ = np.sqrt(x_b**2 + y_b**2 + z_b**2) - np.sqrt(3 * (box_size)**2)
    
    # Ensure the distance is not negative
    if d_ < 0:
        d_ = 0
        
    return d_


def load_snapshot(path_base, c_, mode, Lbox_Mpc, f_mass):
    """
    Loads halo data from a specified path based on the mode.

    :param path_base: Base path to the data
    :param c_: Configuration index
    :param mode: Mode of the data ('rockstar' or other)
    :param f_mass: Mass factor
    Lbox_Mpc: Lbox in Mpc
    :return: Dictionary containing halo data
    """
    c__ = f'{int(c_):03}'
    if mode == 'rockstar':
        m = np.loadtxt(f'{path_base}/_{int(c_)}/halos_0.0.ascii')
        output = {
            'x': m[:, 8],
            'y': m[:, 9],
            'z': m[:, 10],
            'M': np.log10(m[:, 2])
        }
    else:
        p = f'{path_base}run.00{c__}.fofstats.0'
        pkd_halo_dtype = np.dtype([("rPot", ("f4", 3)), ("minPot", "f4"), ("rcen", ("f4", 3)),
                                   ("rcom", ("f4", 3)), ("cvom", ("f4", 3)), ("angular", ("f4", 3)),
                                   ("inertia", ("f4", 6)), ("sigma", "f4"), ("rMax", "f4"),
                                   ("fMAss", "f4"), ("fEnvironDensity0", "f4"),
                                   ("fEnvironDensity1", "f4"), ("rHalf", "f4")])
        halos = np.fromfile(p, count=-1, dtype=pkd_halo_dtype)
        int_fac = 1.0
        halo_center1 = Lbox_Mpc * (halos["rPot"] * int_fac + halos["rcen"] + 0.5)
        halo_center1 = np.array(halo_center1)
        output = {
            'x': halo_center1[:, 0],
            'y': halo_center1[:, 1],
            'z': halo_center1[:, 2],
            'rhalf': Lbox_Mpc * halos["rMax"] * int_fac,
            'M': np.log10((halos['fMAss'] * f_mass).value)
        }
    return output

def read_tipsy(name, offset=0, count=-1):
    """
    Reads out particles from a Tipsy snapshot file.
    
    :param name: Path to the snapshot file
    :param offset: Number of particles to skip at the beginning
    :param count: Number of particles to read, -1 to read all particles
    :return: Header and particles data from the snapshot file
    """
    with open(name, "rb") as f:
        p_header_dt = np.dtype([('a', '>d'), ('npart', '>u4'), ('ndim', '>u4'), 
                                ('ng', '>u4'), ('nd', '>u4'), ('ns', '>u4'), 
                                ('buffer', '>u4')])
        p_header = np.fromfile(f, dtype=p_header_dt, count=1, sep='')
        n_part = ((p_header["buffer"] & 0x000000ff).astype(np.uint64) << 32)[0] + p_header["npart"][0]
        print(f"Total number of particles: {n_part}")

        p_dt = np.dtype([('mass', '>f'), ("x", '>f'), ("y", '>f'), ("z", '>f'),
                         ("vx", '>f'), ("vy", '>f'), ("vz", '>f'), ("eps", '>f'), ("phi", '>f')])
        count = n_part - int(offset) if count == -1 else count
        print(count)
        p = np.fromfile(f, dtype=p_dt, count=int(count), sep='', offset=offset * p_dt.itemsize)

    return p_header, p






def return_params(path_runs, folder, run):
    # Extract the seed from the run identifier
    seed = int(run.split('run')[1])
    
    # Print the run and folder for debugging purposes
    print('run: ', run)
    print('folder: ', folder)

    # Open the file containing the parameters
    with open(path_runs) as f:
        # Initialize lists to store the parameter values
        om_ = []
        h_ = []
        Omega_b_ = []
        n_s_ = []
        sigma_8_ = []
        w_ = []

        # Iterate over each line in the file
        for i, f_ in enumerate(f):
            if i == 0:
                pass  # Skip the header line (if there is one)
            else:
                # Extract and store each parameter from the line
                values = f_.strip().split(',')
                om_.append(float(values[0]))
                sigma_8_.append(float(values[1]))
                w_.append(float(values[2]))
                Omega_b_.append(float(values[3]))
                h_.append(float(values[4]))
                n_s_.append(float(values[5]))

    # Select the parameters corresponding to the seed value
    om = om_[seed - 1]
    sigma_8 = sigma_8_[seed - 1]
    w = w_[seed - 1]
    Omega_b = Omega_b_[seed - 1]
    n_s = n_s_[seed - 1]
    h = h_[seed - 1] * 100. * u.km / u.s / u.Mpc

    # Return the selected parameters
    return om, sigma_8, w, Omega_b, n_s, h



def process_resume(path_z_file):
    # Initialize the resume dictionary with empty lists
    resume = {
        'Step': [],
        'z_far': [],
        'z_near': [],
        'delta_z': [],
        'cmd_far': [],
        'cmd_near': [],
        'delta_cmd': []
    }

    # Open the file containing z values
    with open(path_z_file) as z_fil:
        z = []
        # Iterate over each line in the file
        for z__, z_ in enumerate(z_fil):
            if z__ > 0:
                # Split the line by commas and convert to float
                mute = np.array(z_.split(',')).astype(float)
                
                # Append each value to the corresponding list in the resume dictionary
                resume['Step'].append(mute[0])
                resume['z_far'].append(mute[1])
                resume['z_near'].append(mute[2])
                resume['delta_z'].append(mute[3])
                resume['cmd_far'].append(mute[4])
                resume['cmd_near'].append(mute[5])
                resume['delta_cmd'].append(mute[6])

    # Find the index of the last occurrence of the value 49 in the 'z_far' list
    init = np.where(np.array(resume['z_far']) == 49)[0][-1]

    # Adjust the lists in the resume dictionary from the found index
    resume['Step'] = np.array(resume['Step'])[init:] - init
    resume['z_far'] = np.array(resume['z_far'])[init:]
    resume['z_near'] = np.array(resume['z_near'])[init:]
    resume['delta_z'] = np.array(resume['delta_z'])[init:]
    resume['cmd_far'] = np.array(resume['cmd_far'])[init:]
    resume['cmd_near'] = np.array(resume['cmd_near'])[init:]
    resume['delta_cmd'] = np.array(resume['delta_cmd'])[init:]

    # Return the processed resume dictionary
    return resume




def save_halocatalog(file, max_step_halocatalog, resume, interpolated_distance_to_redshift, mode = 'fof'):
    """
    Save the halo catalog to a FITS file with specified columns and data types,
    including header comments to explain the units.
    
    Parameters:
    file (str): Path to the directory where the halo catalog will be saved.
    max_step_halocatalog (int): Maximum step for the halo catalog.
    interpolated_distance_to_redshift: interp1d object mapping distances (Mpc/h) to redshift
    resume (dict): Dictionary containing resume data.
    """
    

    path_to_save = file + '/halo_catalog.fits'

    # Initialize the final catalog dictionary
    final_cat = {
        'x': [],
        'y': [],
        'z': [],
        'M': [],
        'redshift': [],
        'R': []
    }


    count = 0
    collect = []

    # Iterate through each step in the halo catalog
    for i_ in frogress.bar(np.arange(0, max_step_halocatalog)):
        i = len(resume['Step']) - i_ - 1
        d_min = resume['cmd_near'][i]
        d_max = resume['cmd_far'][i]
        step = resume['Step'][i]

        # Load the snapshot data for the current step
        output_ = load_snapshot(file, step, mode, resume['Lbox_Mpc'], resume['f_mass'])

        number_14 = len(output_['M'][output_['M'] > 14.])
        collect.append(number_14)
        replicas_max = math.ceil(d_max / resume['Lbox_Mpc'] + 1)
        replicas_min = math.ceil(d_min / resume['Lbox_Mpc'] + 1)

        #print('')
        #print('d_max: ',d_max)
        #print('replicas: ',replicas_max) 
        count_i = 0
        add = 0

        f = 1.0
        # Iterate through replicas
        for x_i in range(-replicas_max, replicas_max + 1):
            for y_i in range(-replicas_max - 1, replicas_max + 1):
                for z_i in range(-replicas_max - 1, replicas_max + 1):
                    close_box = _far_point_box(resume['Lbox_Mpc'], x_i, y_i, z_i)

                    if d_min > close_box:
                        if count_i == 0:
                            new_x = output_['x'] + x_i * resume['Lbox_Mpc']
                            new_y = output_['y'] + y_i * resume['Lbox_Mpc']
                            new_z = output_['z'] + z_i * resume['Lbox_Mpc']
                            r = np.sqrt(new_x**2 + new_y**2 + new_z**2)
                            mask = (r >= d_min) & (r < d_max)
                            final_cat_x = new_x[mask]
                            final_cat_y = new_y[mask]
                            final_cat_z = new_z[mask]
                            final_cat_M = output_['M'][mask]
                            final_cat_R = output_['rhalf'][mask]
                            final_cat_redshift = interpolated_distance_to_redshift(r[mask])
                            count_i += 1
                            add += 1
                        else:
                            new_x = output_['x'] + x_i * resume['Lbox_Mpc']
                            new_y = output_['y'] + y_i * resume['Lbox_Mpc']
                            new_z = output_['z'] + z_i * resume['Lbox_Mpc']
                            r = np.sqrt(new_x**2 + new_y**2 + new_z**2)
                            mask = (r >= d_min) & (r < d_max)
                            final_cat_x = np.hstack([final_cat_x, new_x[mask]])
                            final_cat_y = np.hstack([final_cat_y, new_y[mask]])
                            final_cat_z = np.hstack([final_cat_z, new_z[mask]])
                            final_cat_M = np.hstack([final_cat_M, output_['M'][mask]])
                            final_cat_R = np.hstack([final_cat_R, output_['rhalf'][mask]])
                            final_cat_redshift = np.hstack([final_cat_redshift, interpolated_distance_to_redshift(r[mask])])
                            add += 1

        if count == 0:
            if add > 0:
                final_cat['pix_16384_ring'] = hp.pixelfunc.vec2pix(8192 * 2, np.array(final_cat_x), np.array(final_cat_y), np.array(final_cat_z), nest=False).astype('uint32')
                final_cat['M'] = (final_cat_M * 1000).astype('uint16')
                final_cat['redshift'] = (final_cat_redshift * 10000).astype('uint16')
                final_cat['R'] = (final_cat_R * 1000).astype('uint16')
                count += 1
        else:
            final_cat['pix_16384_ring'] = np.hstack([final_cat['pix_16384_ring'], hp.pixelfunc.vec2pix(8192 * 2, np.array(final_cat_x), np.array(final_cat_y), np.array(final_cat_z), nest=False).astype('uint32')])
            final_cat['M'] = np.hstack([final_cat['M'], (final_cat_M * 1000).astype('uint16')])
            final_cat['R'] = np.hstack([final_cat['R'], (final_cat_R * 1000).astype('uint16')])
            final_cat['redshift'] = np.hstack([final_cat['redshift'], (final_cat_redshift * 10000).astype('uint16')])

  

    # Save the final catalog to a FITS file
    if os.path.exists(path_to_save):
        os.remove(path_to_save)

    fits_f = Table()
    fits_f['pix_16384_ring'] = (final_cat['pix_16384_ring']).astype('uint32')
    fits_f['log_M'] = (final_cat['M']).astype('uint16')  # this is Msun/h ---
    fits_f['R'] = (final_cat['R']).astype('uint16')
    fits_f['redshift'] = (final_cat['redshift']).astype('uint16')
    
    hdu = fits.BinTableHDU(data=fits_f)
    
    # Add comments to the header
    hdu.header.add_comment("log_M is in Msun/h units; it needs to be divided by 1000")
    hdu.header.add_comment("redshift needs to be divided by 10000")
    
    # Write the HDU to a FITS file
    hdu.writeto(path_to_save, overwrite=True)

    # Clean up
    del final_cat
    gc.collect()
