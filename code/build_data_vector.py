import numpy as np 
from orphics import stats
import os
import argparse

parser = argparse.ArgumentParser(description="Script to build data vector object.")
parser.add_argument("--input-cosmo", type=str, required=True, help="Path to input cosmology (.txt)")
parser.add_argument("--input-ps", type=str, required=True, help="Path to input lensing power spectra (.txt)")
parser.add_argument("--formatter", type=str, default="#", help="Custom placeholder for index (default: #)")
parser.add_argument("--sim-start", type=int, default=0, help="Starting index for sims.")
parser.add_argument("--sim-end", type=int, default=99, help="Ending index for sims.")
parser.add_argument("--add-noise", action="store_true", help="Add noise according to Gaussian covariance.")
parser.add_argument("--Lmax", type=int, default=2000, help="Maximum L for lensing reconstruction.")
parser.add_argument("--output", type=str, help="Path to write output files. If directory " + \
                    "doesn't exist, the script will create it.")

PATH_TO_NLKK = "/data5/act/release/dr6_lensing_v1/maps/baseline/N_L_kk_act_dr6_lensing_v1_baseline.txt"

args = parser.parse_args()

# god knows why anyone would want different formatters for cosmo / ps files
formatter_count = args.input_ps.count(args.formatter)
full_formatter = args.formatter * formatter_count

bin_edges = np.append([2,6,12,20,30,40,60],  np.arange(80,args.Lmax,80))
binner = stats.bin1D(bin_edges)

bin_cents = 0.5*(bin_edges[1:] + bin_edges[:-1])
delta_bins = bin_edges[1:] - bin_edges[:-1]

if args.add_noise: print("Adding noise realization to data vectors.")
# return noise draw from Gaussian covariance of Clkk + Nlkk
def noise_draw(ells, clkk):
    nlkk = np.loadtxt(PATH_TO_NLKK, usecols=[1])
    if nlkk.size < clkk.size:
       nlkk = np.concatenate((nlkk, np.zeros(clkk.size-nlkk.size)))
    cents, clkknlkk = binner.bin(ells, clkk+nlkk)
    noise_cov = 2/(2*cents+1) * 1/delta_bins * clkknlkk**2

    return np.random.normal(0., noise_cov**0.5) 

data = {}
data['param_labels'] = np.array(['Om', 's8'])
data['params'] = np.zeros((args.sim_end-args.sim_start+1,
                           len(data['param_labels'])))
data['data_vector'] = np.zeros((args.sim_end-args.sim_start+1,
                                len(bin_edges)-1))

filename_cosmo = args.input_cosmo
OmMs, S8s = np.loadtxt(filename_cosmo, unpack=True, usecols=[3,4])

for index in range(args.sim_start, args.sim_end+1):
    filename_ps = args.input_ps.replace(full_formatter,
                                        str(index).zfill(formatter_count))
    try:
        data_ells, data_ps = np.loadtxt(filename_ps, unpack=True) # ells, clkk
    except FileNotFoundError:
        print(f"Couldn't find {filename_ps}, skipping.")
        data['params'] = np.delete(data['params'], (-1), axis=0)
        data['data_vector'] = np.delete(data['data_vector'], (-1), axis=0)
        continue

    cents, bclkk = binner.bin(data_ells, data_ps)
    if args.add_noise:
        bclkk += noise_draw(data_ells, data_ps)
    data['cents'] = cents
    data['data_vector'][index] = np.array(bclkk)
    data['params'][index] = np.array([OmMs[index], S8s[index]])
    print(f"Added {filename_ps}.")

np.save(args.output + ".npy", data)
print(f"Saved data to {args.output}.npy.")

