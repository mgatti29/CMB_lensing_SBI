import numpy as np
import utils
from orphics import maps
import argparse
from pixell import enmap
from solenspipe.utility import get_mask

parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default=None)
parser.add_argument('--qid','-qID',dest='qID',nargs='+',type=str,default=['pa4av4','pa5av4','pa5bv4','pa6av4','pa6bv4'],help='Map IDs, among soapack data/all_arrays*.csv')
parser.add_argument('--prepare_maps', action='store_true',help='load maps')
parser.add_argument('--coadd', action='store_true',help='load coadd (per array/freq) map and ivar')
args = parser.parse_args()
qids=args.qID
LC = utils.LensConfig(args.filepath)
lras,ldecs = np.loadtxt(LC.catalog_large,unpack=True,delimiter=',')
rras,rdecs = np.loadtxt(LC.catalog_regular,unpack=True,delimiter=',')
lcoords = np.asarray((ldecs,lras))
rcoords = np.asarray((rdecs,rras))
lrad = 10.0
rrad = 6.0
mask1 = maps.mask_srcs(LC.shape,LC.wcs,lcoords,lrad)
mask2 = maps.mask_srcs(LC.shape,LC.wcs,rcoords,rrad)
jmask = mask1 & mask2
jmask = ~jmask

mask = get_mask(LC.mask)

for qid in qids:
    array,freq = LC.get_array_freq(qid)
    if args.prepare_maps:
        sig_map = enmap.read_map(f"{LC.d2_data_maps_path}{LC.d2_srcfree_stack_maps_fn%(array,freq)}")
        ivar_map = enmap.read_map(f"{LC.d2_data_maps_path}{LC.d2_ivar_stack_maps_fn%(array,freq)}")
        sig_map[:,:,mask<0.25]=0.0 ##intial maps had been masked (before downgrading) -- maybe delete here --- do we have to mask with new inpainting??
        ivar_map[:,mask<0.25]=0.0
        ip_map = maps.gapfill_edge_conv_flat(sig_map, jmask,ivar=ivar_map) #make sure everything is getting inpainted
        ip_map[:,:,mask<0.25]=0.0
        enmap.write_map(f'{LC.inpainted_path}{LC.inp_srcfree_stack_maps_fn%(array,freq)}',ip_map)
        del ip_map
        del ivar_map
    if args.coadd:
        sig_map = enmap.read_map(f"{LC.d2_data_maps_path}{LC.d2_srcfree_coadd_maps_fn%(array,freq)}")
        print(sig_map.shape)
        ivar_map = enmap.read_map(f"{LC.d2_data_maps_path}{LC.d2_ivar_coadd_maps_fn%(array,freq)}")
        print(ivar_map.shape)
        sig_map[:,mask<0.25]=0.0
        ivar_map[mask<0.25]=0.0
        ip_map = maps.gapfill_edge_conv_flat(sig_map, jmask,ivar=ivar_map) #make sure everything is getting inpainted
        ip_map[:,mask<0.25] = 0.0
        enmap.write_map(f'{LC.inpainted_path}{LC.inp_srcfree_coadd_maps_fn%(array,freq)}',ip_map)
        del ip_map
        del ivar_map