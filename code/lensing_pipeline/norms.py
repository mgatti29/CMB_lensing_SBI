import numpy as np
import argparse
import utils
from solenspipe.utility import get_mask
from falafel import qe
from falafel import utils as futils
from pixell import enmap
import pytempura
import yaml
import solenspipe

parser = argparse.ArgumentParser(description="Script to calculate normalizations in LC")
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument( "--ph", action='store_true',help='Do profile-hardening')
parser.add_argument( "--bh", action='store_true',help='Do point source-hardening')
parser.add_argument( "--est1", type=str, help = "Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.")

args = parser.parse_args()
LC = utils.LensConfig(args.filepath)
mask=get_mask(LC.mask)
ucls,__ = futils.get_theory_dicts(lmax=LC.mlmax,grad=True)
tcls = np.load(f'{LC.filter_path}'+f'{LC.tcl_fndict}',allow_pickle='TRUE').item()
est_list = LC.est_norm_list.copy()

if args.bh:
    e2='src'
    est_list.append(e2)
    est_list.append("TT") # WHY ADD TT AGAIN?? ASK FRANK
    if args.ph:
        profile=np.loadtxt(LC.profile)
        Als = pytempura.get_norms(est_list,ucls,ucls,tcls,LC.lmin,LC.lmax,k_ellmax=LC.mlmax,profile=profile)
    else:
        profile = None
        Als = pytempura.get_norms(est_list,ucls,tcls,LC.lmin,LC.lmax,k_ellmax=LC.mlmax) #why only ucls,tcls (compare to above) ASK FRANK!!!
    R_src_tt = pytempura.get_cross(e2,'TT',ucls,tcls,LC.lmin,LC.lmax,k_ellmax=LC.mlmax,profile=profile)
else:
    Als = pytempura.get_norms(est_list,ucls,ucls,tcls,LC.lmin,LC.lmax,k_ellmax=LC.mlmax)

ls = np.arange(Als[args.est1][0].size)
# Convert to noise per mode on lensing convergence ?? ASK FRANK ABOUT THIS
e1 = args.est1.upper()
Nl_g = Als[e1][0] * (ls*(ls+1.)/2.)**2.
Nl_c = Als[e1][1] * (ls*(ls+1.)/2.)**2.

if args.bh:
    Nl_g_bh = solenspipe.bias_hardened_n0(Als[e1][0],Als[e2],R_src_tt) * (ls*(ls+1.)/2.)**2.
    np.savetxt(f'{LC.norm_path}'+f'{LC.Rsrctt_fn}',R_src_tt)
    np.savetxt(f'{LC.norm_path}'+f'{LC.Nl_g_bh_fn}',Nl_g_bh)
    
np.save(f'{LC.norm_path}'+f'{LC.Als_fn}', Als) 
np.savetxt(f'{LC.norm_path}'+f'{LC.N0g_fn}',Nl_g)
np.savetxt(f'{LC.norm_path}'+f'{LC.N0c_fn}',Nl_c)    
