import numpy as np
import camb
import argparse
import healpy as hp
from pixell import enmap, lensing, curvedsky as cs
from orphics import maps,mpi,stats,io
from pixell import utils as putils
from enlib import bench

# assumes ombh2 = 0.02219218
PATH_TO_CAMB_INI_GPC = "/data5/act/release/dr6_lensing_v1/sim_inputs/theory/cosmo2017_10K_acc3_params.ini"
DEFAULT_MAP_TEMPLATE = "/data5/sims/sbi_project/sim_cmb_lensed_map_fiducial.fits"
DEFAULT_NOISE_UK = 10.
DEFAULT_FWHM_ARCMIN = 1.5

parser = argparse.ArgumentParser(description="Script to make simple mock CMB lensed maps for different cosmologies")
parser.add_argument("outdir",type=str,help="Path where mock maps and parameters will be saved.")
parser.add_argument("--no-maps",action='store_true',help="Skip generating maps for these cosmologies.")
parser.add_argument("--path-to-ini",type=str,default=PATH_TO_CAMB_INI_GPC,help="Path to ini CAMB file.")
parser.add_argument("--add-white-noise",action='store_true',help="Whether to add white-noise to the mock maps.")
parser.add_argument("--white-noise-level",type=float,default=DEFAULT_NOISE_UK,
                    help="White-noise level to be added to maps in uK-arcmin units.")
parser.add_argument("--beam-maps",action="store_true",help="Whether to convolve maps with gaussian beams.")
parser.add_argument("--beam-fwhm",type=float,default=DEFAULT_FWHM_ARCMIN,
                    help="FWHM of beam in arcmin units.")
parser.add_argument("--path-map-template",type=str,default=DEFAULT_MAP_TEMPLATE,
                    help="Path to template map with shape and wcs for the output maps")

args = parser.parse_args()

camb_fn = args.path_to_ini
pars_t = camb.read_ini(camb_fn)

pars_t.WantTransfer = True
results_t = camb.get_results(pars_t)

sigma8_min = 0.6
sigma8_max = 0.9
n_sigma8_steps = 160
sigma8_arr = np.linspace(sigma8_min,sigma8_max,n_sigma8_steps)
omch2_min = 0.05
omch2_max = 0.50
n_omch2_steps = 160
omch2_arr = np.linspace(omch2_min,omch2_max,n_omch2_steps)

r0 = sigma8_arr**2/results_t.get_sigma8()**2
As = pars_t.InitPower.As
As_arr = r0*As

As_grid,omch2_grid = np.meshgrid(As_arr,omch2_arr)
As_grid = As_grid.flatten()
omch2_grid = omch2_grid.flatten()

lmax = 3000
mlmax = 4000
nside = 2048

pars = camb.read_ini(camb_fn) #loading again so that WantTransfer = False again

shape, wcs = enmap.read_map_geometry(args.path_map_template)

as_out = []
omch2_out = []
omegam_out = []
sigma8_out = []
sim_num = []
ps_out = []

nsims = len(As_grid)
comm,rank,my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

if args.no_maps:
    if rank == 0:
        print("Not generating CMB lensing maps.")

for task in my_tasks:
    this_As = As_grid[task]
    this_omch2 = omch2_grid[task]
    with bench.show("Setting pars and calc CAMB results"):
        pars.InitPower.As = this_As
        pars.omch2 = this_omch2
        pars.WantTransfer = True
        this_omegam = pars.omegam
        results = camb.get_results(pars)
        this_sigma8 = results.get_sigma8()[0]

    if (this_sigma8 > sigma8_max) or (this_sigma8 < sigma8_min):
        print(f"Rank {rank}: Rescaled sigma8 = {this_sigma8:0.3f} is out-of-bounds, skipping.")
        continue
    else:
        print(f"Rank {rank}: Rescaled sigma8 = {this_sigma8:0.3f} is in bounds.")

    
    with bench.show("Getting CAMB power spectra"):
        powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
        unlensed_cls = {'tt': powers['unlensed_scalar'][:,0], 'te': powers['unlensed_scalar'][:,3],
                        'ee': powers['unlensed_scalar'][:,1], 'bb': powers['unlensed_scalar'][:,2]}
        lensed_cls = {'tt': powers['lensed_scalar'][:,0], 'te': powers['lensed_scalar'][:,3],
                    'ee': powers['lensed_scalar'][:,1], 'bb': powers['lensed_scalar'][:,2],
                    'pp': powers['lens_potential'][:,0]}
    
        ps = np.array([[unlensed_cls['tt'], unlensed_cls['te'], 0 * unlensed_cls['te']],
                       [unlensed_cls['te'], unlensed_cls['ee'], 0 * unlensed_cls['te']],
                       [0 * unlensed_cls['tt'], 0 * unlensed_cls['te'], 0 * unlensed_cls['bb']]])
        
        ell_ = np.arange(len(lensed_cls['pp']))  # Define ell array for the power spectrum
        kappa_cls = lensed_cls['pp'] * (ell_ * (ell_ + 1) / 2)**2
        
        io.save_cols(f"{args.outdir}sim_cmb_{str(task).zfill(4)}_iclkk.txt",
                     (ell_, kappa_cls))
        
    if not args.no_maps:
        with bench.show("Lensing map"):
                # Generate random alms (spherical harmonic coefficients) for the power spectra
                alms_ = cs.rand_alm(ps, ainfo=None, lmax=lmax, seed=None,
                                    dtype=np.complex128, m_major=True, return_ainfo=False)
                kappa_cmb = hp.synfast(kappa_cls, nside=nside, lmax=mlmax)
                kappa_cmb_alm = hp.map2alm(kappa_cmb,lmax=mlmax)
                ell,emm = hp.Alm.getlm(lmax=mlmax)
                phi_cmb_alm = kappa_cmb_alm / (ell * (ell + 1) / 2)  # Calculate the lensing potential alms
                phi_cmb_alm[ell==0] = 1e-30
                maps_ = lensing.lens_map_curved((3, shape[0], shape[1]), wcs, phi_cmb_alm, alms_,
                                                phi_ainfo=None, maplmax=None, dtype=np.float64, spin=[0, 2], output="l",
                                                geodesic=True, verbose=False, delta_theta=None)
                calms_ = cs.map2alm(maps_[0], lmax=mlmax, spin=[0, 2])
            
        with bench.show("Convolving with beam, adding noise and saving"):
            if args.beam_maps:
                print("convolving with beam")
                calms_ = cs.almxfl(calms_,lambda x: maps.gauss_beam(args.beam_fwhm,x))
            omap = cs.alm2map(calms_,enmap.empty((3,)+shape,wcs,dtype=np.float32),spin=[0,2])
            if args.add_white_noise:
                nmap = maps.white_noise((3,)+shape,wcs,args.white_noise_level)
                nmap[1:] *= np.sqrt(2.)
            else:
                nmap = 0.
            out_map = omap + nmap
            print(shape)
            print(out_map.shape)
            print(wcs)
            print(out_map.wcs)
            enmap.write_map(f"{args.outdir}sim_cmb_lensed_map_{str(task).zfill(4)}.fits",out_map)
    
    sim_num.append(task)
    as_out.append(this_As)
    omch2_out.append(this_omch2)
    omegam_out.append(this_omegam)
    sigma8_out.append(this_sigma8)
    
with io.nostdout():
    s.get_stacks()

with bench.show("MPI Gather"):
    sim_num = putils.allgatherv(sim_num,comm)
    as_out = putils.allgatherv(as_out,comm)
    omch2_out = putils.allgatherv(omch2_out,comm)
    omegam_out = putils.allgatherv(omegam_out,comm)
    sigma8_out = putils.allgatherv(sigma8_out,comm)

if rank == 0:
    io.save_cols(f"{args.outdir}cosmo_params.txt",(sim_num,as_out,omch2_out,omegam_out,sigma8_out))
