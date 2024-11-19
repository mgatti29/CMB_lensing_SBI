import numpy as np
import camb
import argparse
import healpy as hp
from pixell import enmap, lensing, curvedsky as cs
from orphics import maps,mpi,stats,io
from pixell import utils as putils
from enlib import bench

parser = argparse.ArgumentParser(description="Script to make simple mock CMB lensed maps for different cosmologies")
parser.add_argument("--path-to-ini",type=str,help="Path to ini CAMB file.")
parser.add_argument("--add-white-noise",action='store_true',help="Whether to add white-noise to the mock maps.")
parser.add_argument("--white-noise-level",type=float,help="White-noise level to be added to maps in uK-arcmin units.")
parser.add_argument("--beam-maps",action="store_true",help="Whether to convolve maps with gaussian beams.")
parser.add_argument("--beam-fwhm",type=float,help="FWHM of beam in arcmin units.")
parser.add_argument("--path-map-template",type=str,help="Path to template map with shape and wcs for the output maps")
parser.add_argument("--outdir",type=str,help="Path where mock maps and parameters will be saved.")

args = parser.parse_args()

camb_fn = args.path_to_ini
pars_fid = camb.read_ini(camb_fn)

pars_fid.WantTransfer = True
results_fid = camb.get_results(pars_fid)

lmax = 3000
mlmax = 4000
nside= 4096

shape, wcs = enmap.read_map_geometry(args.path_map_template)



powers = results_fid.get_cmb_power_spectra(pars_fid, CMB_unit='muK', raw_cl=True)
unlensed_cls = {'tt': powers['unlensed_scalar'][:,0], 'te': powers['unlensed_scalar'][:,3],
                'ee': powers['unlensed_scalar'][:,1], 'bb': powers['unlensed_scalar'][:,2]}
lensed_cls = {'tt': powers['lensed_scalar'][:,0], 'te': powers['lensed_scalar'][:,3],
            'ee': powers['lensed_scalar'][:,1], 'bb': powers['lensed_scalar'][:,2],
            'pp': powers['lens_potential'][:,0]}

ps = np.array([[unlensed_cls['tt'], unlensed_cls['te'], 0 * unlensed_cls['te']],
                                [unlensed_cls['te'], unlensed_cls['ee'], 0 * unlensed_cls['te']],
                                [0 * unlensed_cls['tt'], 0 * unlensed_cls['te'], 0 * unlensed_cls['bb']]])


# Generate random alms (spherical harmonic coefficients) for the power spectra
alms_ = cs.rand_alm(ps, ainfo=None, lmax=lmax, seed=None, dtype=np.complex128, m_major=True, return_ainfo=False)

ell_ = np.arange(len(lensed_cls['pp']))  # Define ell array for the power spectrum
kappa_cmb = hp.synfast((lensed_cls['pp'] * (ell_ * (ell_ + 1) / 2)**2), nside=nside, lmax=lmax)
kappa_cmb_alm = hp.map2alm(kappa_cmb,lmax=lmax)
ell,emm = hp.Alm.getlm(lmax=lmax)
phi_cmb_alm = kappa_cmb_alm / (ell * (ell + 1) / 2)  # Calculate the lensing potential alms
phi_cmb_alm[ell==0] = 1e-30
maps_ = lensing.lens_map_curved((3, shape[0], shape[1]), wcs, phi_cmb_alm, alms_, phi_ainfo=None, maplmax=None, dtype=np.float64, spin=[0, 2], output="l", geodesic=True, verbose=False, delta_theta=None)
calms_ = cs.map2alm(maps_[0], lmax=lmax, spin=[0, 2])
        

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
enmap.write_map(f"{args.outdir}sim_cmb_lensed_map_fiducial.fits",out_map)
