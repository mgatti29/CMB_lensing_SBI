import numpy as np
import argparse
import utils
from pixell import enmap
from solenspipe.utility import get_mask
import healpy as hp
from pixell import curvedsky as cs
from solenspipe import utility as simgen
from orphics import stats,mpi
from soapack import interfaces as sints
from solenspipe.utility import kspace_mask

parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument("--set",type=int,default=0)
parser.add_argument('--qid','-qID',dest='qID',nargs='+',type=str,default=['pa4av4','pa5av4','pa5bv4','pa6av4','pa6bv4'])
parser.add_argument("--coadd", action='store_true',help='coadd all the splits together')
parser.add_argument("--kvfilter", type=int,  default=90,help="vertical fourier strip width")
parser.add_argument("--khfilter", type=int,  default=50,help="horizontal fourier strip width")
parser.add_argument("--split", type=int,  default=0,help="if coadd=True this is ignored")
parser.add_argument("--lmax", type=int, default = 6000, help = 'lmax of analysis')
args = parser.parse_args()

LC = utils.LensConfig(args.filepath)
qids = args.qID
LMAX=args.lmax
lmax_f = 5400
lmax_out = 5000
specs = ['I','Q','U']
nspecs = len(specs)
nqids = len(qids)
nalms = utils.get_nalms(5000, 5000)
noise_specs = np.zeros((nspecs, nqids, LMAX+1), dtype = np.float64)
#dataModel = getattr(sints,'DR6v4')()

#LOAD MASK
mask=get_mask(LC.mask) #make sure it is downgraded mask d2

#LOAD FOREGROUNDS PER FREQ AND MAKE COVARIANCE
fg_90 = hp.read_alm(f"{LC.fg_path}{LC.fg_f090}")
fg_150 = hp.read_alm(f"{LC.fg_path}{LC.fg_f150}")
cls_fg_90 = simgen.smooth_rolling_cls(cs.alm2cl(fg_90),N=10)
cls_fg_150 = simgen.smooth_rolling_cls(cs.alm2cl(fg_150),N=10)
cls_fg_90x150 = simgen.smooth_rolling_cls(cs.alm2cl(fg_90,fg_150),N=10)
cov= np.zeros((2,2,lmax_f+1))
cov[0,0]=cls_fg_150[:lmax_f+1]
cov[0,1]=cls_fg_90x150[:lmax_f+1]
cov[1,0]=cov[0,1]
cov[1,1]=cls_fg_90[:lmax_f+1]

all_ivars = []
for q in range(len(qids)):
    array,freq = LC.get_array_freq(qids[q])
    ivar_splits = enmap.read_map(f"{LC.d2_data_maps_path}{LC.d2_ivar_stack_maps_fn%(array,freq)}")
    all_ivars.append(ivar_splits)
    for ispec, spec in enumerate(specs):
        noise_specs[ispec,q] = np.loadtxt(f"{LC.kcoadd_path}{LC.kcoadd_noise_weights%(qids[q],spec)}")

#all_ivars = np.array(all_ivars)
#print(all_ivars.shape)
print(type(all_ivars[0][0]))
print(all_ivars[0][0].wcs)

comm,rank,my_tasks = mpi.distribute(LC.nsims_iv)
s = stats.Stats(comm)

#LOAD CALIBRATIONS
arr_dict=   LC.array_dict
cal_dict={}
ells_cal_dict={}
for q in range(len(qids)):
    array,freq = LC.get_array_freq(qids[q])
    ells_cal,cal = np.loadtxt(f"{LC.calibration%(array,freq)}",unpack=True)
    cal = np.interp(np.arange(6000),ells_cal,cal)
    cal_dict[q]=cal
    ells_cal_dict[q]=ells_cal

width = 10

splits = np.arange(LC.nsplits)


for task in my_tasks:
    #if args.set == 0:
    #    noise_i = task
    #elif args.set == 1:
    #    noise_i = task + 400
    setstr = str(args.set).zfill(2)
    simstr = str(task).zfill(4)
    signal = simgen.get_beamed_signal(task,args.set,None,mask.shape,mask.wcs)  #noise already beam deconvolved
    signal = utils.apod(signal,width) #apodize before applying window function in FFT
    wsignal=enmap.apply_window(signal, pow=1.0)
    alms_f=cs.rand_alm(cov,seed=(0,args.set,task+1))
    fg150_i=cs.alm2map(np.array([alms_f[0],alms_f[0]*0.,alms_f[0]*0.]),enmap.empty((3,)+mask.shape,mask.wcs))
    fg90_i=cs.alm2map(np.array([alms_f[1],alms_f[1]*0.,alms_f[1]*0.]),enmap.empty((3,)+mask.shape,mask.wcs))
    
    noiseTot = []
    coadded_alms_specs = np.zeros((nspecs, nqids, nalms), dtype=np.complex128)

    for q in range(len(qids)):
        ivar_list = []
        for j in range(len(splits)):
            ivar_list.append(all_ivars[q][j])
        ivar=ivar_list

        if args.coadd==False:
            fname = f"{LC.noise_sims_path}{LC.noise_sims_split_fn_pattern%(qids[q],str(args.split),simstr)}"
            imap=enmap.read_map(fname)
            alm=cs.map2alm(imap,lmax=5400)
            b_ell = LC.get_beam_function(freq,array,coadd=False,splitnum=args.split)
            alm = cs.almxfl(alm,lambda x: 1/b_ell(x))
            alm[0]=cs.almxfl(alm[0],1/cal)
            alm[1]=alm[1]/LC.pol_eff[qids[q]] 
            alm[2]=alm[2]/LC.pol_eff[qids[q]] 
            noise=cs.alm2map(alm,enmap.empty((3,)+mask.shape, mask.wcs))
            noise[~np.isfinite(noise)] = 0

        else:
            split_stack = []
            for sp in range(len(splits)):
                fname = f"{LC.noise_sims_path}{LC.noise_sims_split_fn_pattern%(qids[q],str(splits[sp]),simstr)}"
                imap = enmap.read_map(fname)
                alm = cs.map2alm(imap,lmax=lmax_f)
                b_ell = LC.get_beam_function(freq,array,coadd=False,splitnum=splits[sp])
                alm = cs.almxfl(alm,lambda x: 1/b_ell(x))
                alm[0]=cs.almxfl(alm[0],1/cal)
                alm[1]=alm[1]/LC.pol_eff[qids[q]] 
                alm[2]=alm[2]/LC.pol_eff[qids[q]] 
                noise=cs.alm2map(alm,enmap.empty((3,)+mask.shape,mask.wcs))
                split_stack.append(noise)
            omap = enmap.zeros(split_stack[0].shape, wcs=split_stack[0].wcs)
            ivareff=enmap.zeros(ivar[0].shape, wcs=ivar[0].wcs)
            split_stack = np.array(split_stack)
            ivar=np.array(ivar)
            for sp in range(LC.nsplits):
                omap+=ivar[sp]*split_stack[sp]
            ivareff = np.sum(ivar,axis=0)
            noise = omap/ivareff
        noise[~np.isfinite(noise)] = 0 #beam deconvolved noise 
        
        if qids[q] in ['pa4av4','pa5bv4','pa6bv4']:
            fg = fg150_i
        else:
            fg = fg90_i
        
        cmb=(noise+wsignal+fg)*mask
        cmb=kspace_mask(cmb,vk_mask=[-1*args.kvfilter,args.kvfilter], hk_mask=[-1*args.khfilter,args.khfilter],deconvolve=True)
        
        noiseTot.append(cmb)
        noise_alms=cs.map2alm(cmb,lmax=5000).astype(np.complex128)
        coadded_alms_specs[0,q] = noise_alms[0]
        coadded_alms_specs[1,q] = noise_alms[1]
        coadded_alms_specs[2,q] = noise_alms[2]
        del ivar_list
        del ivar
        del imap
        del alm
        del noise
        del fg
        del cmb
        del noise_alms

        
    dummy_beams = np.ones(noise_specs[0].shape)
    kcoadd_I = simgen.kspace_coadd(coadded_alms_specs[0], dummy_beams, noise_specs[0])
    kcoadd_Q = simgen.kspace_coadd(coadded_alms_specs[1], dummy_beams, noise_specs[1])
    kcoadd_U = simgen.kspace_coadd(coadded_alms_specs[2], dummy_beams, noise_specs[2])
    kcoadd=cs.alm2map(np.array([kcoadd_I,kcoadd_Q,kcoadd_U]),enmap.empty((3,)+mask.shape,mask.wcs))
    k_alms=cs.map2alm(kcoadd,lmax=5000).astype('complex64')
    if args.coadd == True:
        fname=f"{LC.if_sims_path}{LC.if_sims_fn_pattern_coadd}" % (setstr,simstr)
        
    else:
        fname=f"{LC.if_sims_path}{LC.if_sims_fn_pattern_split}" % (setstr,simstr,args.split)
    hp.write_alm(fname, k_alms,overwrite=True)
    del signal
    del wsignal
    del alms_f
    del fg150_i
    del fg90_i
    del coadded_alms_specs
    del noiseTot
    del dummy_beams
    del kcoadd_I
    del kcoadd_Q
    del kcoadd_U
    del kcoadd
    del k_alms
    

del noise_specs