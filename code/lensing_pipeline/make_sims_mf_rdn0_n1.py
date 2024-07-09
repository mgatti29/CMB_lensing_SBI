import numpy as np
import argparse 
import utils
from orphics import stats,mpi
from pixell import enmap,curvedsky as cs
from solenspipe.utility import get_mask
from solenspipe import utility as simgen
from solenspipe.utility import kspace_mask
import healpy as hp

parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument("--set", type=int,  default=0,help="Set used for the seed in cmb simulation, from 0 to 3")
parser.add_argument("--kvfilter", type=int,  default=90,help="vertical fourier strip width")
parser.add_argument("--khfilter", type=int,  default=50,help="horizontal fourier strip width")
print("starting")

args = parser.parse_args()
LC = utils.LensConfig(args.filepath)

nsims=LC.nsims_rdn0 + 2  #RDN0 REQUIRES LARGEST NUMBER OF SIMS
comm,rank,my_tasks = mpi.distribute(nsims)
s = stats.Stats(comm)

mask=get_mask(LC.mask)
full_res=enmap.read_map(LC.fullres_path)

def apod(imap,width):
    # This apodization is for FFTs. We only need it in the dec-direction
    # since the AdvACT geometry should be periodic in the RA-direction.
    return enmap.apod(imap,[width,0]) 
width=10

for task in my_tasks:
    #used for cmb seeds and naming the files
    setstr = str(args.set).zfill(2)
    simstr = str(task).zfill(4)
    #signal=simgen.get_beamed_signal(task+1,args.set,None,full_res.shape,full_res.wcs)  #noise already beam convolved
    signal=simgen.get_beamed_signal(task,args.set,None,full_res.shape,full_res.wcs)  #noise already beam convolved
    print(signal.shape)
    signal = apod(signal,width) #apodize before applying window function in FFT
    signal=enmap.apply_window(signal, pow=1.0)
    signal=enmap.downgrade(signal,2)
    signal[:,mask<0.25] = 0.0
    kmap=kspace_mask(signal,vk_mask=[-1*args.kvfilter,args.kvfilter], hk_mask=[-1*args.khfilter,args.khfilter],deconvolve=True) 
    k_alms=cs.map2alm(kmap,lmax=5000).astype('complex64')
    fname=f"{LC.mf_sims_path}{LC.mf_sims_fn_pattern}" % (setstr,simstr)
    hp.write_alm(fname, k_alms,overwrite=True)
    del signal
    del kmap
    del k_alms

del mask
