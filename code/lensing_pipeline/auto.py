import numpy as np
import utils 
import argparse
from solenspipe.utility import get_mask
from enlib import bench
from pixell import curvedsky as cs
from pixell import enmap
from solenspipe.utility import w_n
from solenspipe import four_split_phi,split_phi_to_cl
from pixell import lensing as plensing
import healpy as hp
from falafel import utils as futils
from pixell import utils as putils
from orphics import io, mpi, stats
import gc

parser = argparse.ArgumentParser(description="Script to calculate auto spectrum")
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument( "--ph", action='store_true',help='Do profile-hardening')
parser.add_argument( "--bh", action='store_true',help='Do point source-hardening')
parser.add_argument( "--est1", type=str, help = "Estimator 1, one of TT,TE,EE,EB,TB,MV,MVPOL.")
parser.add_argument( "--nsets", type=int,  default=2,help="Number for sets for mean-field script")
parser.add_argument( "--dataseed", type=int,  default=0,help="Data seeds")
parser.add_argument( "--blind", action='store_true',help='apply random blind factor to map')
parser.add_argument( "--model_subtract", action='store_true',help='remove tSZ template.')
#parser.add_argument("--final_2splits",action='store_true',help="Final number of splits") # DO WE NEED ANOTHER ARGUMENT LIKE FRANK'S twosplits (boolean) -- semed redundant in pipeline

args = parser.parse_args()
e1 = args.est1.upper()
LC = utils.LensConfig(args.filepath)
mask=get_mask(LC.mask)

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
if args.model_subtract:
    tszsub = True
else:
    tszsub = False
with bench.show("get_data"):
    Xdat={}
    for split in splits:
        Xdat[split] = LC.get_data(split,tszsub)

if LC.nsplits == 4:
    powfunc = lambda x,y,m,cross,ikalm: split_phi_to_cl(x,y,m,cross,ikalm)
    phi_names=['phi_xy_X','phi_xy01','phi_xy02','phi_xy03','phi_xy12','phi_xy13','phi_xy23','phi_xy_x0','phi_xy_x1','phi_xy_x2','phi_xy_x3']
    xy=four_split_phi(Xdat[0],Xdat[1],Xdat[2],Xdat[3],q_func1=qfunc)
elif LC.nsplits == 1:
    powfunc = lambda x,y,m,cross,ikalm: utils.phi_to_cl(x,y,m,cross,ikalm)
    phi_names=['phi_xy00']
    xy=np.array([plensing.phi_to_kappa(qfunc(Xdat[0],Xdat[0]))])
else:
    print("No implementation for other splits")
xy_g=[]
xy_c=[]
for i in range(len(xy)):
    xy_g.append(xy[i][0])
    xy_c.append(xy[i][1])
xy_grad=np.array(xy_g)
xy_curl=np.array(xy_c)

with bench.show("Read and subtract mean-field"):
    mf,mfc = LC.get_meanfield(phi_names,args.nsets,e1)
    xy_grad_mf = {}
    xy_curl_mf = {}
    for s in range(args.nsets):
        xy_grad_mf[s] = xy_grad - mf[s]
        xy_curl_mf[s] = xy_curl - mfc[s]
    
with bench.show("Blinding"):
    if args.blind:
        np.random.seed(1997)
        blind_factor=np.random.uniform(low=0.9, high=1.1)
        for s in range(args.nsets):
            xy_grad_mf[s] = blind_factor*xy_grad_mf[s]
            xy_curl_mf[s] = blind_factor*xy_curl_mf[s]

with bench.show("save mf subtracted"):
    
    if LC.nsplits == 4:
        nfactor = 6
        coadd_alm = xy_grad_mf[0][1:7]  #these numbers 1:7 might have to change if different data ??
        coaddc_alm = xy_curl_mf[0][1:7]
    elif LC.nsplits == 1:
        nfactor = 1
        coadd_alm = xy_grad_mf[0]
        coaddc_alm = xy_curl_mf[0]
    else:
        coadd_alm = None
        coaddc_alm = None
        
    coadd_alm=np.sum(coadd_alm,axis=0)/nfactor
    coaddc_alm=np.sum(coaddc_alm,axis=0)/nfactor

    if args.blind:
        blinding = "True"
    else:
        blinding = "False"
    ofn = f'{LC.ps_path}{LC.ps_coaddgrad_fn}'%(LC.nsplits,args.est1,blinding)
    ofnc = f'{LC.ps_path}{LC.ps_coaddcurl_fn}'%(LC.nsplits,args.est1,blinding)

#nsims_mf = LC.nsims_mf
nruns = 1
comm,rank,my_tasks = mpi.distribute(nruns)
stat = stats.Stats(comm)
uicls = []
autog_cls  = []
autoc_cls  = []
inputx_cls = []

for task in my_tasks:
    with bench.show("get ikalm"):
        sim_id = task
        ikalm = futils.get_kappa_alm(sim_id).astype(np.complex128)
        ikalm=cs.map2alm(cs.alm2map(ikalm,enmap.empty((1,)+LC.shape,LC.wcs))[0]*mask,lmax=LC.mlmax)
        uicl = cs.alm2cl(ikalm,ikalm)/w_n(mask,2)
        uicls.append(uicl)    

    with bench.show("auto and cross spectrum"):
        ### REWRITE SPLIT_TO_CL FUNCTION IN SOLENSPIPE
        ## AUTO ##
        autog = powfunc(xy_grad_mf[0],xy_grad_mf[1],m=LC.nsplits,cross=False,ikalm=None)/w_n(mask,4)
        autoc = powfunc(xy_curl_mf[0],xy_curl_mf[1],m=LC.nsplits,cross=False,ikalm=None)/w_n(mask,4)
        #mcl_bh = powfunc(mf[0],mf[1],m=LC.nsplits,cross=False,ikalm=None)/w_n(mask,4)
        #mcurlcl_bh = powfunc(mfc[0],mfc[1],m=LC.nsplits,cross=False,ikalm=None)/w_n(mask,4)
        ## CROSS ##
        inputx = powfunc(xy_grad,xy_grad,m=LC.nsplits,cross=True,ikalm=ikalm)/w_n(mask,3)
        #I don't think line below gets calculated in franks-irene's code so i'm commenting out here
        #uxcl_bh_curl = powfunc(xy_c,xy_c,m=LC.nsplits,cross=True,ikalm=ikalm)/w_n(mask,3)
        autog_cls.append(autog)
        autoc_cls.append(autoc)
        inputx_cls.append(inputx)
        del ikalm, autog, autoc, inputx
        gc.collect()

with io.nostdout():
    stat.get_stacks()

with bench.show("MPI gather"):
    uicls  = putils.allgatherv(uicls, comm)
    autog_cls  = putils.allgatherv(autog_cls, comm)
    autoc_cls  = putils.allgatherv(autoc_cls, comm)
    inputx_cls = putils.allgatherv(inputx_cls, comm)

if rank == 0:
    with bench.show("Save"):
        np.save(f'{LC.ps_path}{LC.ps_icl_fn}'%(e1),uicls)
        np.save(f'{LC.ps_path}{LC.ps_xcl_fn}'%(e1),inputx_cls)
        np.save(f'{LC.ps_path}{LC.ps_autograd_fn}'%(args.bh,e1,blinding),autog_cls)
        np.save(f'{LC.ps_path}{LC.ps_autocurl_fn}'%(args.bh,e1,blinding),autoc_cls)
        np.save(f'{LC.ps_path}{LC.ps_xygradmf_fn}'%(args.est1,blinding),xy_grad_mf)
        hp.write_alm(ofn,coadd_alm,overwrite=True)
        hp.write_alm(ofnc,coaddc_alm,overwrite=True)
        if args.blind:
            np.savetxt(f'{LC.ps_path}{LC.ps_blind_factor}',np.ones(1)*blind_factor)
