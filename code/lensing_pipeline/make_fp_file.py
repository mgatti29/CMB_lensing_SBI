import argparse 
import numpy as np
from orphics import io
import yaml
import re

parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default=None)
#parser.add_argument('--simnums','-simnums',dest='simnums',nargs='+',type=int,default=[1980,1981,1982,1983,1984,1985,1985,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999])

args = parser.parse_args()
fpaths = io.config_from_yaml(args.filepath)

simnums = np.arange(1980,1999,1)
for s in simnums:
    nn = str(s).zfill(5)
    fn_out = re.sub(".yaml",f"_{nn}.yaml",args.filepath)
    print(fn_out)
    data_maps_fn_pattern = f"sim_cmb_night_%s_%s_{nn}_3pass_1way_set%s_map.fits"
    fpaths["data_maps_fn_pattern"] = data_maps_fn_pattern
    kcoadded_alms = f"kcoadd_data_tszsub_{nn}_%s_%s.fits" 
    fpaths["kcoadded_alms"] = kcoadded_alms
    ps_coaddgrad_fn = f"coadd_{nn}_%ssplitlensingmap%s_blind%s.fits"
    ps_coaddcurl_fn = f"coaddc_{nn}_%ssplitlensingmap%s_blind%s.fits"
    ps_icl_fn = f"icl_{nn}_%s.npy"
    ps_xcl_fn = f"xcl_{nn}_%s.npy"
    ps_autograd_fn =f"auto_{nn}_bh%s_%s_blind%s.npy"
    ps_autocurl_fn = f"autocurl_{nn}_bh%s_%s_blind%s.npy"
    ps_xygradmf_fn = f"xygrad_{nn}%sblind%s.npy"
    ps_blindfactor_fn = f"blindfactor_{nn}.txt"
    fpaths["ps_coaddgrad_fn"] = ps_coaddgrad_fn
    fpaths["ps_coaddcurl_fn"] = ps_coaddcurl_fn
    fpaths["ps_icl_fn"] = ps_icl_fn
    fpaths["ps_xcl_fn"] = ps_xcl_fn
    fpaths["ps_autograd_fn"] = ps_autograd_fn
    fpaths["ps_autocurl_fn"] = ps_autocurl_fn
    fpaths["ps_xygradmf_fn"] = ps_xygradmf_fn
    fpaths["ps_blindfactor_fn"] = ps_blindfactor_fn
    rdn0_fn = f"rdn0_{nn}_bh%s_est%s_s%s.npy"
    mcn0_fn = f"mcn0_{nn}_bh%s_est%s_s%s.npy"
    fpaths["rdn0_fn"] = rdn0_fn
    fpaths["mcn0_fn"] = mcn0_fn
    file=open(fn_out,"w")
    yaml.dump(fpaths,file,sort_keys=False)
    file.close()


