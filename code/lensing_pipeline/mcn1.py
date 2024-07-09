import numpy as np
import utils
import argparse
from orphics import mpi
from solenspipe.utility import get_mask, w_n
from falafel import utils as futils
from pixell import curvedsky as cs
from enlib import bench
from solenspipe import bias

parser = argparse.ArgumentParser(description="Script to calculate realization dependent N0/Gaussian bias")
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument( "--ph", action='store_true',help='Do profile-hardening')
parser.add_argument( "--bh", action='store_true',help='Do point source-hardening')
parser.add_argument( "--est1", type=str, help = "Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.")

args = parser.parse_args()
e1 = args.est1.upper()
LC = utils.LensConfig(args.filepath)
comm,rank,my_tasks = mpi.distribute(1)
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
mask = get_mask(LC.mask)

tcls = np.load(f'{LC.filter_path}'+f'{LC.tcl_fndict}',allow_pickle='TRUE').item()
ucls,tcls = futils.get_theory_dicts(nells=tcls,lmax=LC.mlmax,grad=True)

with bench.show("QFUNC"):
    if args.bh:
        if args.ph:
            profile=np.loadtxt(LC.profile)
        else:
            profile=None
    else:
        profile=None
    qfunc = LC.make_q_func(args.bh,ucls,e1,profile)

powfunc = lambda x,y: cs.alm2cl(x,y)

with bench.show("make get_kmap functions"):
    def get_kmap(seed):
        icov = seed[0]
        set = seed[1]
        nsim = seed[2]
        talm,ealm,balm = LC.load_mf_sim_iter(set=set,nsim=nsim)
        Xdat = np.array([talm,ealm,balm])
        return Xdat
    
mcn1 = bias.mcn1(0,get_kmap,powfunc,LC.nsims_n1,qfunc,qfunc2=qfunc,comm=comm)
ells = np.arange(LC.mlmax+1) #CHECK IF THIS IS ALWAYS THE CASE
mcn1 = mcn1 * (ells*(ells+1.)/2.)**2./w_n(mask,4)
if rank==0: 
    np.save(f'{LC.n1_path}{LC.n1_fn}'%(str(args.bh),str(e1)),mcn1)