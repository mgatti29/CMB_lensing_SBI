'''
downgrade:
    downgrade ivar and srcfree maps by factor 2
'''
import argparse
import gc
import os
import numpy as np
from pixell import enmap
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default="filepaths.txt")
parser.add_argument('--qid','-qID',dest='qID',nargs='+',type=str,default=['pa4av4','pa5av4','pa5bv4','pa6av4','pa6bv4'],help='Map IDs, among soapack data/all_arrays*.csv')
parser.add_argument('--prepare_maps', action='store_true',help='load maps and downgrade by 2')
parser.add_argument('--prepare_ivars', action='store_true',help='load ivars and downgrade by 2, save per-split')
parser.add_argument('--coadd', action='store_true',help='load coadd (per array/freq) map and ivar, downgrade')

args = parser.parse_args()
qids=args.qID

LC = utils.LensConfig(args.filepath)

os.makedirs(LC.d2_data_maps_path, exist_ok=True)

# downgrade is simple:
splits = np.arange(LC.nsplits)

for qid in qids:
    array,freq = LC.get_array_freq(qid)
    if args.prepare_maps:
        ivars = LC.get_mapstack_for_preproc(array,freq,"ivar",calibrated=True)
        print(ivars.shape)
        ivars_d2 = enmap.downgrade(ivars, 2, op = np.sum)
        print(ivars_d2.shape)
        enmap.write_map(f'{LC.d2_data_maps_path}{LC.d2_ivar_stack_maps_fn%(array,freq)}', ivars_d2)
        del ivars
        gc.collect()
        srcfrees = LC.get_mapstack_for_preproc(array,freq,"src_free",calibrated=True)
        print(srcfrees.shape)
        srcfrees_d2 = enmap.downgrade(srcfrees, 2)
        print(srcfrees_d2.shape)
        enmap.write_map(f'{LC.d2_data_maps_path}{LC.d2_srcfree_stack_maps_fn%(array,freq)}',srcfrees_d2)
        del srcfrees_d2
        gc.collect()

        if args.prepare_ivars:
            ivars_d2=enmap.read_map(f'{LC.d2_data_maps_path}{LC.d2_ivar_stack_maps_fn%(array,freq)}')
            print(ivars_d2.shape)
            for i in range(len(ivars_d2)):
                fname=f'{LC.d2_data_maps_path}{LC.d2_ivar_split_maps_fn%(array,freq,str(i))}'
                print(ivars_d2[i].shape)
                enmap.write_map(fname,ivars_d2[i])
            del ivars_d2
            gc.collect()

    if args.coadd:
        print("COADD ---------------")
        ivar_coadd,fn = LC.get_ivar_for_preproc(array,freq,coadd=True,calibrated=False) #why are we not calibrating these???
        print(ivar_coadd.shape)
        ivar_coadd_d2 = enmap.downgrade(ivar_coadd, 2, op = np.sum)
        print(ivar_coadd_d2.shape)
        enmap.write_map(f'{LC.d2_data_maps_path}{LC.d2_ivar_coadd_maps_fn%(array,freq)}', ivar_coadd_d2)
        del ivar_coadd_d2
        gc.collect()
        srcfree_coadd,fn = LC.get_srcfree_for_preproc(array,freq,coadd=True,calibrated=False) 
        print(srcfree_coadd.shape)
        srcfree_coadd_d2 = enmap.downgrade(srcfree_coadd, 2)
        print(srcfree_coadd_d2.shape)
        enmap.write_map(f'{LC.d2_data_maps_path}{LC.d2_srcfree_coadd_maps_fn%(array,freq)}',srcfree_coadd_d2)
        del srcfree_coadd_d2
        gc.collect()

