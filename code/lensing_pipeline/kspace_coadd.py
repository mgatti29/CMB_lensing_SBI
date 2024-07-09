import numpy as np
import argparse
from pixell import enmap
import utils
from solenspipe.utility import get_mask,kspace_mask
from solenspipe import utility as simgen
from scipy.interpolate import interp1d
from orphics import maps,io
from pixell import curvedsky as cs
import healpy as hp
from pixell import enplot

parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument('--qid','-qID',dest='qID',nargs='+',type=str,default=['pa4av4','pa5av4','pa5bv4','pa6av4','pa6bv4'])
parser.add_argument("--model_subtract", action='store_true',help='remove tSZ template.')
parser.add_argument("--coadd", action='store_true',help='coadd all the splits together')
parser.add_argument("--recalculate_weights", action='store_true',help='recalculate kweights used')
parser.add_argument("--kvfilter", type=int,  default=90,help="vertical fourier strip width")
parser.add_argument("--khfilter", type=int,  default=50,help="horizontal fourier strip width")
parser.add_argument("--split", type=int,  default=0,help="if coadd=True this is ignored")
parser.add_argument("--lmax", type=int, default = 6000, help = 'lmax of analysis')

args = parser.parse_args()
qids = args.qID

data_run = False

LC = utils.LensConfig(args.filepath)
mask = get_mask(LC.mask)

sz_nemo = {}
if args.model_subtract:
    sz90=enmap.read_map(LC.nemomodel_f090)
    sz_nemo["f090"] = sz90
    sz150=enmap.read_map(LC.nemomodel_f150)
    sz_nemo["f150"] = sz150

#load beams
splits = np.arange(LC.nsplits)
LMAX = args.lmax

all_maps = []
all_ivars = []
beam_fns = []
specs = ['I','Q','U']
nspecs = len(specs)
nqids = len(qids)
nalms = utils.get_nalms(LMAX, LMAX)
noise_specs = np.zeros((nspecs, nqids, LMAX+1), dtype = np.float64)
coadded_alms_specs = np.zeros((nspecs, nqids, nalms), dtype=np.complex128)



for q in range(len(qids)):
    array,freq = LC.get_array_freq(qids[q])
    beam_q = LC.get_beam_function(freq,array,coadd=True)  #Irene has a renorm=True param but only for day maps
    beam_fns.append(beam_q)    
    map_splits = enmap.read_map(f'{LC.inpainted_path}{LC.inp_srcfree_stack_maps_fn%(array,freq)}')
    ivar_splits = enmap.read_map(f"{LC.d2_data_maps_path}{LC.d2_ivar_stack_maps_fn%(array,freq)}")
    #enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/map_splits",enplot.plot(map_splits[0]))
    if args.model_subtract:
        sz_beam = LC.get_sz_beams(freq)
        foreground = simgen.reconvolve_maps(sz_nemo[freq],mask,sz_beam,beam_q)
        for j in range(len(map_splits)):
            map_splits[j][0] = map_splits[j][0] - foreground #only subtract foreground from T map
    all_maps.append(map_splits)
    all_ivars.append(ivar_splits)
    dec_maps = [] #deconvolved beam, pixell window and kspace filter
    dec_ivars = [] #assoc ivars of decon maps (ivars are not deconvolved)
    #enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/all_maps0",enplot.plot(all_maps[0]))

    for sp in range(LC.nsplits):
        if data_run:
            this_beam = LC.get_beam_function(freq,array,splitnum=sp) #use this if running with real data
            smap = utils.deconvolve_maps(map_splits[sp],mask,this_beam,lmax=6000)
        else:
            alm =cs.map2alm(map_splits[sp],lmax=6000)
            alm_decon = cs.almxfl(alm,lambda ell:1/utils.gauss_beam(ell,1.4))
            imap = enmap.empty((3,)+mask.shape,mask.wcs,dtype=np.float32)
            smap = cs.alm2map(alm_decon,imap)
        #enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/smaps0",enplot.plot(smap[0]))
        dmap = kspace_mask(smap,vk_mask=[-1*args.kvfilter,args.kvfilter], hk_mask=[-1*args.khfilter,args.khfilter],deconvolve=True)
        #io.plot_img(dmap,down=8,filename=f"{LC.kcoadd_path}kmask_{q}_{sp}.png")
        #enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/dmaps_deconTrue0",enplot.plot(dmap[0]))
        dec_maps.append(dmap)
        dec_ivars.append(ivar_splits[sp])
    
    #print(np.array(dec_maps).shape)
    #print(np.array(dec_ivars).shape)

    if args.recalculate_weights:
        bls=interp1d(np.arange(LMAX),np.ones(LMAX),bounds_error=False,fill_value=0)
        for ispec,spec in enumerate(specs):
            noisecl=utils.get_datanoise(dec_maps,dec_ivars, ispec, ispec, mask,bls,beam_deconvolve=False,N=1)
            bin_edges = np.linspace(2,len(noisecl),300).astype(int)
            cents,cls=simgen.bandedcls(noisecl,bin_edges)
            cls=maps.interp(cents,cls)(np.arange(len(noisecl)))
            noise_specs[ispec, q] = cls
            np.savetxt(f"{LC.kcoadd_path}{LC.kcoadd_noise_weights%(qids[q],spec)}",cls)
    
    else:
        for ispec, spec in enumerate(specs):
            noise_specs[ispec, q] = np.loadtxt(f"{LC.kcoadd_path}{LC.kcoadd_noise_weights%(qids[q],spec)}")
    
    ells_cal, cal = np.loadtxt(f"{LC.calibration%(array,freq)}",unpack=True)
    cal = np.interp(np.arange(LMAX),ells_cal,cal)
    
    if args.coadd == False:
        noise = dec_maps[args.split]
    else:
        dec_splits = dec_maps
        imap = enmap.zeros(dec_splits[0].shape,wcs=dec_splits[0].wcs)
        ivarreff = enmap.zeros(dec_ivars[0].shape,wcs=dec_ivars[0].wcs)
        for j in range(len(dec_ivars)):
            imap += dec_ivars[j]*dec_splits[j]
            ivarreff += dec_ivars[j]
        noise = imap/ivarreff

    noise[~np.isfinite(noise)] = 0
    #io.plot_img(noise[0],down=8,filename=f"{LC.kcoadd_path}noiseT_{q}.png")
    #noise[~np.isnan(noise)] = 0 #I added this just in case
    #enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/noise0",enplot.plot(noise[0]))
    alms=cs.map2alm(noise,lmax=LMAX)
    #imap = enmap.empty(dec_maps[0].shape,wcs=dec_maps[0].wcs)
    #omap = cs.alm2map(alms,imap)
    almsTcal=cs.almxfl(alms[0],1/cal)
    almsQcal=alms[1]/LC.pol_eff[qids[q]]
    almsUcal=alms[2]/LC.pol_eff[qids[q]]
    #imap = enmap.empty(dec_maps[0].shape,wcs=dec_maps[0].wcs)
    #omap = cs.alm2map(almsTcal,imap)
    coadded_alms_specs[0,q]=almsTcal
    coadded_alms_specs[1,q]=almsQcal
    coadded_alms_specs[2,q]=almsUcal

dummy_beam = np.ones(noise_specs[0].shape) # Map already beam-deconvolved
f_shape = all_maps[0][0][0].shape
f_wcs = all_maps[0][0][0].wcs

kcoadd_I = simgen.kspace_coadd(coadded_alms_specs[0], dummy_beam, noise_specs[0])
kcoadd_Q = simgen.kspace_coadd(coadded_alms_specs[1], dummy_beam, noise_specs[1])
kcoadd_U = simgen.kspace_coadd(coadded_alms_specs[2], dummy_beam, noise_specs[2])
kcoadd = cs.alm2map(np.array([kcoadd_I, kcoadd_Q, kcoadd_U]), enmap.empty((3,) + f_shape, f_wcs))
#imap = enmap.empty(f_shape,f_wcs)
#omap = cs.alm2map(kcoadd_I,imap)
#io.plot_img(omap,down=8,filename=f"{LC.kcoadd_path}kcoadd_I.png")
#enplot.write("/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/kcoadd0",enplot.plot(kcoadd[0]))

#this doesn't work :()
#Ealm,Balm=simgen.pureEB(kcoadd[1],kcoadd[2],mask,returnMask=0,lmax=LMAX,isHealpix=False)
#a=cs.alm2map(np.array([kcoadd_I,Ealm,Balm]),enmap.empty((3,)+f_shape,f_wcs))
k_alms=cs.map2alm(kcoadd,lmax=5000).astype('complex64')
#k_alms=k_alms.astype(np.complex128)  ##This breaks my code!!!!
print(k_alms.shape)
if args.model_subtract:
    tszsub = True
else:
    tszsub = False
if args.coadd:
    coadd_type = "coadded"
else:
    coadd_type = f"{args.split}"

outfn = f"{LC.kcoadd_path}{LC.kcoadded_alms%(tszsub,coadd_type)}"
hp.write_alm(outfn,k_alms,overwrite=True)

print("DONE ------")

