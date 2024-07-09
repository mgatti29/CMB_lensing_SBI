import numpy as np
from pixell import enmap
import healpy as hp
from pixell import curvedsky as cs
from pixell import enplot
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--simnum",type=int,default=2000)
args = parser.parse_args()

### NOT A PART OF THE PIPELINE -- JUST CREATING SIM DATA

"""
(PA4 f150) and f220 (PA4 f220) bands; ##f220 data not included in Qu et al 2023 analysis
PA5 in the f090 (PA5 f090) and f150 (PA5 150) bands, 
and PA6 in the f090 (PA6 f090) and f150 (PA6 f150)
"""

def gauss_beam(ell,fwhm):
    tht_fwhm= np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))

def white_noise(shape,wcs,seed=None,div=None):
    np.random.seed(seed)
    return np.random.standard_normal(shape) / np.sqrt(div)

outdir = "/home/s/sievers/kaper/scratch/lenspipe/sim_run/maps/"
beam_fwhm = 1.4
mask_path = "/gpfs/fs0/project/r/rbond/jiaqu/actpol/DR6_plus/masks/act_mask_fejer1_20220316_GAL070_rms_70.00_downgrade_None.fits"
mask = enmap.read_map(mask_path)
full_shape,full_wcs = mask.shape,mask.wcs
simroot = "/scratch/r/rbond/msyriac/data/sims/alex/v0.4/"
nn = str(args.simnum).zfill(5)
simname = f"{simroot}fullskyLensedUnabberatedCMB_alm_set01_{nn}.fits"
sigalm = hp.read_alm(simname,hdu=(1,2,3))
sigalm_conv = cs.almxfl(sigalm,lambda ell:gauss_beam(ell,beam_fwhm))
imap = enmap.empty((3,)+full_shape,full_wcs,dtype=np.float32)
sigmap_conv = cs.alm2map(sigalm_conv,imap)
sigmap_conv[:,mask<0.25] = 0
sigmap_conv[np.isnan(sigmap_conv)] = 0

m = 'night'
a_f = ['pa4_f150','pa5_f090','pa5_f150','pa6_f090','pa6_f150']

for i,af in enumerate(a_f):
    froot = "/home/s/sievers/kaper/scratch/maps/dr6v4_20230316/" #'/home/s/sievers/kaper/scratch/maps/dr6v3_20211031/'
    fname = f"{froot}cmb_{m}_{af}_3pass_4way_coadd_ivar.fits" #f"{froot}cmb_{m}_{a}_{f}_8way_coadd_ivar.fits"
    ivar = enmap.read_map(fname)
    seed = int(i)+5
    wn_map = white_noise((3,)+full_shape,full_wcs,seed = seed,div=ivar)
    totmap = (sigmap_conv+wn_map)
    totmap[:,mask<0.25] = 0 
    totmap[np.isnan(totmap)] = 0
    enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{nn}_3pass_1way_set0_ivar.fits",ivar)
    enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{nn}_3pass_1way_coadd_ivar.fits",ivar)
    enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{nn}_3pass_1way_set0_map_srcfree.fits",totmap)
    enmap.write_map(f"{outdir}sim_cmb_{m}_{af}_{nn}_3pass_1way_coadd_map_srcfree.fits",totmap)
