import numpy as np
from solenspipe import utility as simgen
from solenspipe.utility import w_n
from orphics import io,maps
import healpy as hp
from pixell import curvedsky as cs
from pixell import enmap
from falafel import qe
import re
from solenspipe.utility import get_mask
import solenspipe
from orphics import mpi
from pixell.mpi import FakeCommunicator
from pixell import lensing as plensing
from pixell import utils as putils
from pixell import enplot

def gauss_beam(ell,fwhm):
    tht_fwhm= np.deg2rad(fwhm/60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))


def get_nalms(lmax, mmax = None):
    '''
    Calculate number of alms given (ell max, m max) [healpy format]
    '''

    if mmax is None:
        mmax = lmax # m = ell
    return int(mmax * (2 * lmax + 1 - mmax) / 2 + lmax + 1)

def get_datanoise(map_list,ivar_list, a, b, mask,beam,N=20,beam_deconvolve=True,lmax=6000):
    ### THIS IS A FUNCTION FROM SOLENSPIPE UTILITY WITH MINOR MODIFICATIONS
    """
    Calculate the noise power of a coadded map given a list of maps and list of ivars.
    Inputs:
    map_list: list of source free splits
    ivar_list: list of the inverse variance maps splits
    a: 0,1,2 for I,Q,U respectively
    b:0,1,2 for I,Q,U, respectively
    N: window to smooth the power spectrum by in the rolling average.
    mask: apodizing mask

    Output:
    1D power spectrum accounted for w2 from 0 to 10000
    """
    
    pmap=enmap.pixsizemap(map_list[0].shape,map_list[0].wcs)


    cl_ab=[]
    n = len(map_list)
    #calculate the coadd maps
    if a!=b:
        coadd_a=simgen.coadd_mapnew(map_list,ivar_list,a)
        coadd_b=simgen.coadd_mapnew(map_list,ivar_list,b)
    else:
        coadd_a=simgen.coadd_mapnew(map_list,ivar_list,a)

    for i in range(n):
        print(i)
        if a!=b:
            d_a=map_list[i][a]-coadd_a
            #noise_a=d_a*mask
            noise_a=d_a #noise already masked
            alm_a=cs.map2alm(noise_a,lmax=lmax)
            d_b=map_list[i][b]-coadd_b
            noise_b=d_b
            alm_b=cs.map2alm(noise_b,lmax=lmax)
            cls = hp.alm2cl(alm_a,alm_b)
            cl_ab.append(cls)
        else:
            d_a=map_list[i][a]-coadd_a

            noise_a=d_a
            #enplot.write(f"/home/s/sievers/kaper/scratch/lenspipe/sim_run/kcoadd/d_{a}_0",enplot.plot(noise_a))    
            print("generating alms")
            alm_a=cs.map2alm(noise_a,lmax=lmax)
            alm_a=alm_a.astype(np.complex128)
            if beam_deconvolve:
                alm_a = cs.almxfl(alm_a,lambda x: 1/beam(x)) 
            cls = hp.alm2cl(alm_a)
            cl_ab.append(cls)
    cl_ab=np.array(cl_ab)
    #sqrt_ivar=np.sqrt(ivar_eff(0,ivar_list))

    mask=mask
    mask[mask<=0]=0
    w2=np.sum((mask**2)*pmap) /np.pi / 4.
    print(w2)
    if n == 1:  #MY MODIFICATION FOR A 1-SPLIT RUN
        power = np.sum(cl_ab, axis=0)
    else:
        power = 1/n/(n-1) * np.sum(cl_ab, axis=0)
    ls=np.arange(len(power))
    power[~np.isfinite(power)] = 0
    power=simgen.rolling_average(power, N)
    bins=np.arange(len(power))
    power=maps.interp(bins,power)(ls)

    return power / w2

def smooth_pack(alms,mask,n):
    cltt = simgen.smooth_cls(hp.alm2cl(alms[0])/w_n(mask,n))
    clee=simgen.smooth_cls(hp.alm2cl(alms[1])/w_n(mask,n)) #this is signal+noise
    clbb=simgen.smooth_cls(hp.alm2cl(alms[2])/w_n(mask,n))
    clte=simgen.smooth_cls(hp.alm2cl(alms[0],alms[1])/w_n(mask,n))
    return np.array([cltt,clee,clbb,clte])

def reshape_alm(fname,mask,lmax):
    alm = hp.read_alm(fname,hdu=(1,2,3))
    pmap= cs.alm2map(alm,enmap.empty((3,)+mask.shape,mask.wcs))*mask
    oalms= cs.map2alm(pmap,lmax=lmax)
    oalms[~np.isfinite(oalms)] = 0
    oalms=oalms.astype(np.complex128)
    return oalms

def deconvolve_maps(maps,mask,beam,lmax=6000):
    "deconvolve the beam of a map" 
    "function from solenspipe.utility (simgen) but slightly modified mask application"
    shape=maps.shape
    wcs=maps.wcs
    maps[:,mask<0.25]=0
    alm_a=cs.map2alm(maps,lmax=lmax)
    alm_a = cs.almxfl(alm_a,lambda x: 1/beam(x)) 
    reconvolved_map=cs.alm2map(alm_a,enmap.empty(shape,wcs))
    return reconvolved_map

def iv_filter(alms,mask,lmin,lmax,mlmax,filter_fn):
    cltt,clee,clbb,clte=np.loadtxt(filter_fn)
    ls=np.arange(len(cltt))
    nells_T = maps.interp(ls,cltt) 
    nells_E= maps.interp(ls,clee)
    nells_B= maps.interp(ls,clbb)
    filt_t = 1./(nells_T(ls))
    filt_e = 1./(nells_E(ls))
    filt_b = 1./(nells_B(ls))
    almt = qe.filter_alms(alms[0].copy(),filt_t,lmin=lmin,lmax=lmax)
    alme = qe.filter_alms(alms[1].copy(),filt_e,lmin=lmin,lmax=lmax)
    almb = qe.filter_alms(alms[2].copy(),filt_b,lmin=lmin,lmax=lmax)
    return almt,alme,almb

def phi_to_cl(xy,uv,m=1,cross=False,ikalm=None):
    if cross:
        cl = cs.alm2cl(xy[0],ikalm)
    else:
        cl = cs.alm2cl(xy[0],uv[0])
    return cl

def apod(imap,width):
    # This apodization is for FFTs. We only need it in the dec-direction
    # since the AdvACT geometry should be periodic in the RA-direction.
    return enmap.apod(imap,[width,0]) 

def mcrdn0(set, get_kmap, power, phifunc, nsims, qfunc1, qfunc2=None, Xdat=None,use_mpi=True, 
         verbose=True, skip_rd=False,power_mcn0=None):
    """
    my version of the solenspipe.bias.mcrdn0 function -- no seed stuff
    """
    qa = phifunc
    qf1 = qfunc1 
    qf2 = qfunc2

    mcn0evals = []
    if not(skip_rd): 
        rdn0evals = []

    if use_mpi:
        comm,rank,my_tasks = mpi.distribute(nsims)
    else:
        comm,rank,my_tasks = FakeCommunicator(), 0, range(nsims)
        

    for i in my_tasks:
        if rank==0 and verbose: print("MCRDN0: Rank %d doing task %d" % (rank,i))
        Xs  = get_kmap(set,i)
        if not(skip_rd): 
            qaXXs = qa(Xdat,Xs,qf1)
            qbXXs = qa(Xdat,Xs,qf2) if qf2 is not None else qaXXs 
            qaXsX = qa(Xs,Xdat,qf1)  #split 1
            qbXsX = qa(Xs,Xdat,qf2) if qf2 is not None else qaXsX 
            rdn0_only_term = power(qaXXs,qbXXs) + power(qaXXs,qbXsX) \
                    + power(qaXsX,qbXXs) + power(qaXsX,qbXsX)
            print(rdn0_only_term == 0.0)
        Xsp = get_kmap(set,i+1)  #changed from original code in so-lenspipe Xsp = get_kmap((icov,1,i)) 
        
        if power_mcn0 is None:
            qm = qa
            powerm = power
        else:
            qm = lambda X,Y,qf: plensing.phi_to_kappa(qf(X,Y))
            powerm = power_mcn0
        qaXsXsp = qm(Xs,Xsp,qf1) #split1 
        qbXsXsp = qm(Xs,Xsp,qf2) if qf2 is not None else qaXsXsp #split2
        qaXspXs = qm(Xsp,Xs,qf1)
        qbXspXs = qm(Xsp,Xs,qf2) if qf2 is not None else qaXspXs #this is not present

        mcn0_term = (powerm(qaXsXsp,qbXsXsp) + powerm(qaXsXsp,qbXspXs))
        mcn0evals.append(mcn0_term.copy())
        if not(skip_rd):  rdn0evals.append(rdn0_only_term - mcn0_term)

    if not(skip_rd):
        avgrdn0 = putils.allgatherv(rdn0evals,comm)
    else:
        avgrdn0 = None
    avgmcn0 = putils.allgatherv(mcn0evals,comm)
    return avgrdn0, avgmcn0

def mcrdn0_s4(set, split, get_kmap, power,phifunc, nsims, qfunc1, qfunc2=None, Xdat=None,Xdat1=None,Xdat2=None,Xdat3=None, use_mpi=True, 
         verbose=True, skip_rd=False,shear=False,power_mcn0=None):
    "my version of bias.mcrdn0_s4 bias.mcrdn0_s41 bias.mcrdn0_s42 and bias.mcrdn0_s43 "
    "FIX !!"
    return None
    
class LensConfig:
    def __init__(self,filepaths=None):
        fpaths = io.config_from_yaml(filepaths)
        self.mask = fpaths["mask"]
        mask = get_mask(self.mask)
        self.nsplits = fpaths["nsplits"] #or sets ?? confused
        self.shape,self.wcs = mask.shape,mask.wcs
        self.mlmax = fpaths["mlmax"]
        self.nside = None
        self.px = qe.pixelization(shape=self.shape,wcs=self.wcs,nside=self.nside)
        self.res_arcmin = np.rad2deg(enmap.pixshape(self.shape, self.wcs)[0])*60.
        self.data_maps_path = fpaths["data_maps_path"]
        self.data_maps_fn_pattern = fpaths["data_maps_fn_pattern"]
        self.d2_data_maps_path = fpaths["d2_data_maps_path"]
        self.d2_ivar_stack_maps_fn = re.sub("_set%s_map","_ivar_d2",self.data_maps_fn_pattern)
        self.d2_ivar_split_maps_fn = re.sub("_set%s_map","_set%s_ivar_d2",self.data_maps_fn_pattern)
        self.d2_srcfree_stack_maps_fn = re.sub("_set%s_map","_map_srcfree_d2",self.data_maps_fn_pattern)
        self.inp_srcfree_stack_maps_fn = fpaths["inp_srcfree_stack_maps_fn"]%self.d2_srcfree_stack_maps_fn
        self.d2_ivar_coadd_maps_fn = re.sub("_set%s_map","_coadd_ivar_d2",self.data_maps_fn_pattern)
        self.d2_srcfree_coadd_maps_fn = re.sub("_set%s_map","_coadd_map_srcfree_d2",self.data_maps_fn_pattern)
        self.inp_srcfree_coadd_maps_fn = fpaths["inp_srcfree_coadd_maps_fn"]%self.d2_srcfree_coadd_maps_fn
        self.catalog_regular = fpaths["catalog_regular"]  #EXTERNAL
        self.catalog_large = fpaths["catalog_large"]      #EXTERNAL
        self.fullres_path = fpaths["fullres_path"]
        self.inpainted_path = fpaths["inpainted_path"]
        self.if_sims_path = fpaths["if_sims_path"]
        self.if_sims_fn_pattern_coadd = fpaths["if_sims_fn_pattern_coadd"]
        self.if_sims_fn_pattern_split = fpaths["if_sims_fn_pattern_split"]
        self.nemomodel_f090 = fpaths["nemomodel_f090"]    #EXTERNAL
        self.nemomodel_f150 = fpaths["nemomodel_f150"]    #EXTERNAL
        self.szbeam150 = fpaths["szbeam150"] #check if using correct szbeams   #EXTERNAL
        self.szbeam90 = fpaths["szbeam90"] #checl if using correct szbeams     #EXTERNAL
        self.beams_path = fpaths["beams_path"]             #EXTERNAL
        self.kcoadd_path = fpaths["kcoadd_path"]
        self.kcoadd_noise_weights = fpaths["kcoadd_noise_weights"]
        self.nsims_mf = fpaths["nsims_mf"]
        self.fg_f150 = fpaths["fg_f150"]
        self.fg_f090 = fpaths["fg_f090"]
        self.noise_sims_path = fpaths["noise_sims_path"]  #OG sims -- external
        self.noise_sims_split_fn_pattern = fpaths["noise_sims_split_fn_pattern"]
        self.array_dict = {'pa4av4':'pa4_f150','pa5av4':'pa5_f090','pa5bv4':'pa5_f150','pa6av4':'pa6_f090','pa6bv4':'pa6_f150'}
        self.pol_eff = {'pa4av4':0.9584,'pa5av4':0.9646,'pa5bv4':0.9488,'pa6av4':0.9789,'pa6bv4':0.9656}  #these shouldn't be hardcoded ---CHECK !!!
        self.gain_dict = {     #these shouldn't be hardcoded ---CHECK !!!
		"pa4_f150": 0.9708,
		"pa4_f220": 1.1119,
		"pa5_f090": 0.9625,
		"pa5_f150": 0.9961,
		"pa6_f090": 0.9660,
		"pa6_f150": 0.9764,
	    }
        self.mf_sims_path = fpaths["mf_sims_path"]
        self.mf_sims_fn_pattern = fpaths["mf_sims_fn_pattern"]
        self.calibration = fpaths["calibration"]           #EXTERNAL
        self.kcoadded_alms = fpaths["kcoadded_alms"]
        self.nsims_iv = fpaths["nsims_iv"]
        self.filter_path = fpaths["filter_path"]
        self.filter_fnnpy = fpaths["filter_fnnpy"]
        self.filter_fntxt = fpaths["filter_fntxt"]
        self.tcl_fndict = fpaths["tcl_fndict"]
        self.profile = fpaths["profile"]                    #EXTERNAL
        self.lmin = fpaths["lmin"]
        self.lmax = fpaths["lmax"]
        self.est_norm_list = ['TT','TE','TB','EB','EE','MV','MVPOL']
        self.norm_path = fpaths["norm_path"]
        self.Rsrctt_fn = fpaths["Rsrctt_fn"]%(self.lmin,self.lmax)#f'R_src_tt{self.lmin}_lmax{self.lmax}.txt'
        self.Nl_g_bh_fn = fpaths["Nl_g_bh_fn"]%(self.lmin,self.lmax) #f'Nl_g_bh{self.lmin}_lmax{self.lmax}.txt'
        self.Als_fn = fpaths["Als_fn"]%(self.lmin,self.lmax)#f'Als_lmin{self.lmin}_lmax{self.lmax}.npy'
        self.N0g_fn = fpaths["N0g_fn"]%(self.lmin,self.lmax)#f'N0g_lmin{self.lmin}_lmax{self.lmax}t.txt'
        self.N0c_fn = fpaths["N0c_fn"]%(self.lmin,self.lmax)#f'N0c_lmin{self.lmin}_lmax{self.lmax}.txt'
        #self.data_fn_pattern = fpaths["data_fn_pattern_CHANGE NAME"] #change name BECAUSE CONFUSING
        self.mf_path = fpaths["mf_path"]
        self.mf_grad_fn = fpaths["mf_grad_fn"]
        self.mf_curl_fn = fpaths["mf_curl_fn"]
        self.ps_path = fpaths["ps_path"]
        self.ps_coaddgrad_fn = fpaths["ps_coaddgrad_fn"]
        self.ps_coaddcurl_fn = fpaths["ps_coaddcurl_fn"]
        self.ps_icl_fn = fpaths["ps_icl_fn"]
        self.ps_xcl_fn = fpaths["ps_xcl_fn"]
        self.ps_autograd_fn = fpaths["ps_autograd_fn"]
        self.ps_autocurl_fn = fpaths["ps_autocurl_fn"]
        self.ps_xygradmf_fn = fpaths["ps_xygradmf_fn"]
        self.ps_blindfactor_fn = fpaths["ps_blindfactor_fn"]
        self.fg_path = fpaths["fg_path"]
        self.nsims_rdn0 = fpaths["nsims_rdn0"]
        self.rdn0_path = fpaths["rdn0_path"]
        self.rdn0_fn = fpaths["rdn0_fn"]
        self.mcn0_fn = fpaths["mcn0_fn"]
        self.nsims_n1 = fpaths["nsims_n1"]
        self.n1_path = fpaths["n1_path"]
        self.n1_fn = fpaths["n1_fn"]

    def get_array_freq(self,qid):
        array_freq = self.array_dict[qid]
        array = array_freq[:3]
        freq = array_freq[4:]
        return array,freq
    
    def get_ivar_for_preproc(self,array,freq,sset=None,coadd=False,calibrated=False):  #decide what qid is going to be
        #TO DO raise a warning if uncal map requested
        if coadd==False:
            ivarfn = re.sub("_map","_ivar",self.data_maps_fn_pattern)%(array,freq,str(sset))
        else:
            ivarfn = re.sub("set%s_map","coadd_ivar",self.data_maps_fn_pattern)%(array,freq)
        fn = self.data_maps_path + ivarfn
        key = f"{array}_{freq}"
        if calibrated:
            mul = 1./self.gain_dict[key]
        else:
            mul = 1.
        omap = enmap.read_map(fn) * mul
        #if coadd:
            #omap = enmap.enmap(np.stack(omap),omap.wcs) 
        return omap,fn
    
    def get_srcfree_for_preproc(self,array,freq,sset=None,coadd=False,calibrated=False):
        if coadd==False:
            srcfreefn = re.sub("_map","_map_srcfree",self.data_maps_fn_pattern)%(array,freq,str(sset))
        else:
            srcfreefn = re.sub("set%s_map","coadd_map_srcfree",self.data_maps_fn_pattern)%(array,freq)
        fn = self.data_maps_path + srcfreefn
        key = f"{array}_{freq}"
        if calibrated:
            mul = 1./self.gain_dict[key]**2  #MAKE SURE GAIN IS SQUARED!!
        else:
            mul = 1.
        omap = enmap.read_map(fn) * mul
        #if coadd:
            #omap = enmap.enmap(omap,omap.wcs) 
        return omap,fn
            
    
    def get_mapstack_for_preproc(self,array,freq,mtype,calibrated=False):
        splits = np.arange(self.nsplits)
        stack = []
        for s in range(len(splits)):
            if mtype =="ivar":
                omap,fn = self.get_ivar_for_preproc(array,freq,str(splits[s]),coadd=False,calibrated=calibrated)
            elif mtype == "src_free":
                omap,fn = self.get_srcfree_for_preproc(array,freq,str(splits[s]),coadd=False,calibrated=calibrated)
            stack.append(omap)
        smap = enmap.enmap(np.stack(stack),omap.wcs)
        return smap
    
    def get_sz_beams(self,freq):
        if freq=="f150":
            this_beam = self.szbeam150
        elif freq=="f090":
            this_beam = self.szbeam90
        ls,bells = np.loadtxt(this_beam,unpack=True,usecols=[0,1])
        assert ls[0]==0
        bells = bells/bells[0]
        from scipy.interpolate import interp1d
        return interp1d(ls,bells,bounds_error=False,fill_value=0)
    
    def get_beam_function(self,freq,array,coadd=True,splitnum=None):
        if coadd:
            fn = f"{self.beams_path}coadd_{array}_{freq}_night_beam_tform_jitter_cmb.txt"
        else:
            if splitnum!=None:
                fn = f"{self.beams_path}set{splitnum}_{array}_{freq}_night_beam_tform_jitter_cmb.txt"
        ls,bells = np.loadtxt(fn,unpack=True,usecols=[0,1])
        assert ls[0]==0
        bells = bells/bells[0]
        from scipy.interpolate import interp1d
        return interp1d(ls,bells,bounds_error=False,fill_value=0)


    def get_data(self,split,model_subtract): #maybe add coadd option ?? #check if downgraded maps or og
        data_fn = f'{self.kcoadd_path}{self.kcoadded_alms%(model_subtract,str(split))}'
        mask = get_mask(self.mask)
        alms = reshape_alm(data_fn,mask,self.mlmax)
        filter_fn = f'{self.filter_path}'+f'{self.filter_fntxt}'
        oalms = iv_filter(alms,mask,self.lmin,self.lmax,self.mlmax,filter_fn)
        return oalms
    
    def get_meanfield(self,phi_names,nsets,est1):
        mf={}
        mfc={}
        for s in range(nsets):
            mf[s]=[]
            mfc[s]=[]
        for i in range(len(phi_names)):
            for s in range(nsets):
                mfalm = hp.read_alm(f'{self.mf_path}{self.mf_grad_fn}'%(s,phi_names[i],est1,self.lmin,self.lmax,self.nsims_mf))
                mfalmc = hp.read_alm(f'{self.mf_path}{self.mf_curl_fn}'%(s,phi_names[i],est1,self.lmin,self.lmax,self.nsims_mf))
                mf[s].append(mfalm)
                mfc[s].append(mfalmc)
        for s in range(nsets):
            mf[s] = np.array(mf[s])
            mfc[s] = np.array(mfc[s])
        return mf,mfc
    
            
    def load_iv_filter_sim(self,nsim):
        fname = self.if_sims_path + self.if_sims_fn_pattern_coadd%("00",str(nsim).zfill(4))
        mask = get_mask(self.mask)
        alms = reshape_alm(fname,mask,self.mlmax)
        cls = smooth_pack(alms,mask,2)
        return cls
    
    def load_norms(self,bh):
        Als = np.load(f'{self.norm_path}{self.Als_fn}',allow_pickle='TRUE').item()
        Nl_g = np.loadtxt(f'{self.norm_path}{self.N0g_fn}')
        Nl_c = np.loadtxt(f'{self.norm_path}{self.N0c_fn}')
        if bh:
            Nl_g_bh = np.loadtxt(f'{self.norm_path}'+f'{self.Nl_g_bh_fn}')
            R_src_tt = np.loadtxt(f'{self.norm_path}'+f'{self.Rsrctt_fn}')
        else:
            Nl_g_bh = None
            R_src_tt = None 
        return Als,Nl_g,Nl_c,Nl_g_bh,R_src_tt
    
    def load_mf_sim_iter(self,set,nsim):
        filter_fn = f'{self.filter_path}'+f'{self.filter_fntxt}'
        mask = get_mask(self.mask)
        fname = self.mf_sims_path +self.mf_sims_fn_pattern%(str(set).zfill(2),str(nsim).zfill(4))  # where does split come in CHECK-- ASK FRANK/IRENE!!!!
        mf_sim = reshape_alm(fname,mask,self.mlmax)
        talm,ealm,balm = iv_filter(mf_sim,mask,self.lmin,self.lmax,self.mlmax,filter_fn)
        return talm,ealm,balm
    
    def make_q_func(self,bh,ucls,e1,profile):
        Als,Nl_g,Nl_c,Nl_g_bh,R_src_tt = self.load_norms(bh)
        if bh:
            e2 = 'SRC'
        else:
            e2 = None
            Als['src'] = None
            Als['TT'] = None #check if this is ok because TT key might be duplicated ??
        qfunc = solenspipe.get_qfunc(self.px,ucls,self.mlmax,e1,Al1=Als[e1],est2=e2,Al2=Als['src'],Al3=Als['TT'],R12=R_src_tt,profile=profile)
        return qfunc