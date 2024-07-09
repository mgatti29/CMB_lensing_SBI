import numpy as np
import utils
import argparse
from enlib import bench
from falafel import utils as futils
from solenspipe import four_split_phi,split_phi_to_cl
from pixell import curvedsky as cs
from pixell import lensing as plensing
from solenspipe.utility import get_mask, w_n
from orphics import mpi

parser = argparse.ArgumentParser(description="Script to calculate realization dependent N0/Gaussian bias")
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument( "--ph", action='store_true',help='Do profile-hardening')
parser.add_argument( "--bh", action='store_true',help='Do point source-hardening')
parser.add_argument( "--est1", type=str, help = "Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.")
parser.add_argument( "--set", type=int,  default=0,help="Set of sims")
parser.add_argument( "--model_subtract", action='store_true',help='remove tSZ template.')

args = parser.parse_args()
e1 = args.est1.upper()
LC = utils.LensConfig(args.filepath)
comm,rank,my_tasks = mpi.distribute(1)
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
mask = get_mask(LC.mask)
tcls = np.load(f'{LC.filter_path}'+f'{LC.tcl_fndict}',allow_pickle='TRUE').item()
ucls,tcls = futils.get_theory_dicts(nells=tcls,lmax=LC.mlmax,grad=True)
splits = np.arange(LC.nsplits)



with bench.show("QFUNC"):
    if args.bh:
        if args.ph:
            profile=np.loadtxt(LC.profile)
        else:
            profile=None
    else:
        profile=None
    qfunc = LC.make_q_func(args.bh,ucls,e1,profile)

if LC.nsplits == 4:
    powfunc = lambda x,y: split_phi_to_cl(x,y)
    phifunc = lambda Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0,Xdatp_1,Xdatp_2,Xdatp_3,qf:four_split_phi(Xdat_0,Xdat_1,Xdat_2,Xdat_3,Xdatp_0,Xdatp_1,Xdatp_2,Xdatp_3,qf)
elif LC.nsplits ==1:
    powfunc = lambda x,y: utils.phi_to_cl(x,y)
    phifunc = lambda Xdat_0,Xdatp_0,qf:np.array([plensing.phi_to_kappa(qf(Xdat_0,Xdatp_0))])
powfunc_mcn0 = lambda x,y: cs.alm2cl(x,y)

if args.model_subtract:
    tszsub = True
else:
    tszsub = False

with bench.show("Get data"):
    Xdat={}
    for split in splits:
        Xdat[split] = LC.get_data(split,tszsub)

with bench.show("make get_kmap functions"):
    def get_kmap(set,nsim):
        talm,ealm,balm = LC.load_mf_sim_iter(set=set,nsim=nsim)
        Xdat = np.array([talm,ealm,balm])
        return Xdat
    
with bench.show("cal rnd0 and mc0"):

    if LC.nsplits == 4:
        rdn0,mcn0 = utils.mcrdn0_s4(args.set, args.split, get_kmap, powfunc,phifunc, LC.nsims_rdn0, 
                                    qfunc, qfunc2=None,Xdat=Xdat[0],Xdat1=Xdat[1],Xdat2=Xdat[2],Xdat3=Xdat[3],
                                    use_mpi=True, skip_rd=False,power_mcn0=powfunc_mcn0) 
    elif LC.nsplits ==1 :
        rdn0,mcn0 = utils.mcrdn0(args.set, get_kmap, powfunc, phifunc, LC.nsims_rdn0, qfunc,qfunc2=None,Xdat=Xdat[0],
                                use_mpi=True,skip_rd=False,power_mcn0=powfunc_mcn0)

rdn0 = rdn0/w_n(mask,4)
mcn0 = mcn0/w_n(mask,4)

print("finished rdn0")
print("finished mcn0")

if rank==0:
    np.save(f"{LC.rdn0_path}{LC.rdn0_fn}"%(str(args.bh),str(e1),str(args.set)),rdn0)
    np.save(f"{LC.rdn0_path}{LC.mcn0_fn}"%(str(args.bh),str(e1),str(args.set)),mcn0)

print("done saving rdn0 and mcn0")