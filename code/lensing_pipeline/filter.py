import numpy as np
import argparse
from solenspipe.utility import get_mask
from pixell import utils as putils
from orphics import mpi,stats,io
import utils
from enlib import bench

#LOAD MASK
parser = argparse.ArgumentParser()
parser.add_argument('--filepath',type=str,default=None)
args = parser.parse_args()

LC = utils.LensConfig(args.filepath)
mask=get_mask(LC.mask)

comm,rank,my_tasks = mpi.distribute(LC.nsims_iv)
s = stats.Stats(comm)

filters = []
for task in my_tasks:
    cls_iv_sim = LC.load_iv_filter_sim(task)
    filters.append(cls_iv_sim)
    
with io.nostdout():
    s.get_stacks()
    
with bench.show("MPI Gather"):
    filters = putils.allgatherv(filters,comm)

if rank==0:
    filters=np.mean(filters,axis=0)
    tcls={}
    tcls['TT'] = filters[0][:LC.mlmax+1]
    tcls['TE'] = filters[-1][:LC.mlmax+1]
    tcls['EE'] = filters[1][:LC.mlmax+1]
    tcls['BB'] = filters[2][:LC.mlmax+1]
    np.save(f'{LC.filter_path}'+f'{LC.filter_fnnpy}',filters)
    np.savetxt(f'{LC.filter_path}'+f'{LC.filter_fntxt}',filters)
    np.save(f'{LC.filter_path}'+f'{LC.tcl_fndict}',tcls)
    
    
    
    