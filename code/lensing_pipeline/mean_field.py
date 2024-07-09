import numpy as np
import argparse
import utils
from orphics import mpi,stats,io
import solenspipe
from enlib import bench
from solenspipe.utility import get_mask
from solenspipe import four_split_phi
import healpy as hp
from pixell import lensing as plensing
from falafel import utils as futils

parser = argparse.ArgumentParser(description="Script to calculate mean-field")
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument( "--ph", action='store_true',help='Do profile-hardening')
parser.add_argument( "--bh", action='store_true',help='Do point source-hardening')
parser.add_argument( "--est1", type=str, help = "Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.")
parser.add_argument( "--set", type=int,  default=0,help="Set of sims")
#parser.add_argument("--nsplits", type=int,default=1,help="Number of splits")

args = parser.parse_args()
e1 = args.est1.upper()
LC = utils.LensConfig(args.filepath)
comm,rank,my_tasks = mpi.distribute(LC.nsims_mf)

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
        
splits = np.arange(LC.nsplits)
if LC.nsplits == 4:
    phi_names=['phi_xy_X','phi_xy01','phi_xy02','phi_xy03','phi_xy12','phi_xy13','phi_xy23','phi_xy_x0','phi_xy_x1','phi_xy_x2','phi_xy_x3']
else:
    phi_names=['phi_xy00']
s = stats.Stats(comm)
for task in my_tasks:
    with bench.show("get data"):
        Xdat = {}
        for split in splits:
            #split actually doesnt matter --- it is all the same
            talm,ealm,balm = LC.load_mf_sim_iter(set=args.set,nsim=task)
            Xdat[split] = np.array([talm,ealm,balm])
        if LC.nsplits == 4:
            xy=four_split_phi(Xdat[0],Xdat[1],Xdat[2],Xdat[3],q_func1=qfunc)
        elif LC.nsplits == 1:
            xy=np.array([plensing.phi_to_kappa(qfunc(Xdat[0],Xdat[0]))])
        else:
            print("No implementation for other splits")
        xy_a=[]
        xy_c=[]
        for i in range(len(xy)):
            xy_a.append(xy[i][0])  #DON'T REALLY UNDERSTAND THIS -- ASK FRANK
            xy_c.append(xy[i][1])
        xy_a=np.array(xy_a)
        xy_c=np.array(xy_c)
        
        for i in range(len(phi_names)):
            s.add_to_stack('r'+phi_names[i]+'f',xy_a[i].real)
            s.add_to_stack('i'+phi_names[i]+'f',xy_a[i].imag)
            s.add_to_stack('r'+phi_names[i]+'fc',xy_c[i].real)
            s.add_to_stack('i'+phi_names[i]+'fc',xy_c[i].imag)

            
            
with io.nostdout():
    s.get_stacks()
    
if rank==0:
    for i in range(len(phi_names)):
        mfalm=s.stacks['r'+phi_names[i]+'f'] + 1j*s.stacks['i'+phi_names[i]+'f']
        mfalmc=s.stacks['r'+phi_names[i]+'fc'] + 1j*s.stacks['i'+phi_names[i]+'fc']
        print(type(mfalm))
        print(type(mfalmc))
        hp.write_alm(f'{LC.mf_path}{LC.mf_grad_fn}'%(args.set,phi_names[i],args.est1,LC.lmin,LC.lmax,LC.nsims_mf),mfalm,overwrite=True)  #changed name convention slightly
        hp.write_alm(f'{LC.mf_path}{LC.mf_curl_fn}'%(args.set,phi_names[i],args.est1,LC.lmin,LC.lmax,LC.nsims_mf),mfalmc,overwrite=True)