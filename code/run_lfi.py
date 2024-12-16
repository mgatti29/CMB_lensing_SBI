# based on code by Marco Gatti
# https://github.com/mgatti29/LFI_desy3/blob/main/demo_compression_LFI.ipynb

import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys
import argparse
import warnings
import copy

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam

import pydelfi.priors as priors
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
import emcee as mc

import getdist
from getdist import plots, MCSamples

parser = argparse.ArgumentParser(description="Script to run pydelfi on " + \
                                 "input CMB lensing data vectors.")
parser.add_argument("--input", type=str, required=True,
                    help="Path to input CMB lensing data vectors, " + \
                         "formatted as a .npy file")
parser.add_argument("--output", type=str, required=True,
                    help="Path to write output files. If directory " + \
                         "doesn't exist, the script will create it.")
parser.add_argument("--data-index", type=int,
                    help="Sim index to treat as target data vector.")
parser.add_argument("--input-cosmo", type=str,
                    default="/data6/sims/sbi_outputs_12092024/s8_200_omch2_200_cosmo_params.txt",
                    help="Path to input file with cosmological parameters.")

args = parser.parse_args()

CLIP_MIN = -0.5
CLIP_MAX = 1.5
CLIP_MID = 0.5

DATA_INDEX = 0 if args.data_index is None else args.data_index
RESULTS_DIR = args.output + '/tests/results/'

data = np.load(args.input, allow_pickle=True).item()
try:
    assert data['param_labels'].size > 0
    assert data['params'].size > 0
    assert data['data_vector'].size > 0
except KeyError:
    warnings.warn("Error: Ensure that input data has a 'params', " + \
                  "'param_labels', and 'data_vector' entry.")
    sys.exit()

# output directory
if not os.path.exists(args.output):
    os.mkdir(args.output)

# skip compression, but save normalized data anyway

# normalize data
data_zs = (data['data_vector']-np.median(data['data_vector'],axis=0)) 
data_zs /= np.std(data['data_vector'], axis=0)

# just choosing this scheme based on hyperparameters
sim_data = np.clip(CLIP_MID + (CLIP_MID * 0.4) * data_zs, CLIP_MIN, CLIP_MAX)

# loop over parameters
full_data = {}
full_data['sim_params'] = np.copy(data['params']) # parameters for simulations
full_data['normalized_data_sims'] = np.copy(data_zs) # data vectors for our simulations
full_data['target'] = full_data['normalized_data_sims'][DATA_INDEX]

# LFI
lower = np.array([0.1, 0.6])
upper = np.array([0.5, 0.9])
theta2d_expected_mean = [0.3, 0.75]
prior = priors.Uniform(lower, upper)

def initial_parameters(theta, relative_sigma):
    """
    :param theta: list/array of parameter values
    :param relative_sigma: controls variance of random draws
    :return: the theta array but with random shifts
    """
    theta = np.array(theta)
    return np.random.normal(theta, np.abs(theta * relative_sigma))
     
count = 0
nn = len(lower)
# any time you re-run it, please increase this by the number of NDEs.
base = count 
count += 2
n_data = full_data['normalized_data_sims'].shape[1]
NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=nn,
                                                 n_data=n_data,
                                                 n_hiddens=[50,50],
                                                 n_mades=2,
                                                 act_fun=tf.tanh,
                                                 index=base),
        ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=nn,
                                                 n_data=n_data,
                                                 n_hiddens=[50,50],
                                                 n_mades=3,
                                                 act_fun=tf.tanh,
                                                 index=base + 1)]

pn = ['p{0}'.format(i) for i in range(nn)]

#os.makedirs(RESULTS_DIR, exist_ok=True)
#os.makedirs(RESULTS_DIR + '/' + str(base), exist_ok=True)
#os.system('rm ' + RESULTS_DIR + '/' + str(base) + '/*')
os.makedirs(RESULTS_DIR, exist_ok=True)
try:
    os.mkdir(RESULTS_DIR+'/'+str(base))
except:
    pass
try:
    os.system('rm '+RESULTS_DIR+'/'+str(base)+'/*')
except:
    pass

# abbreviations
full_data_sim_array = np.array(full_data['normalized_data_sims'],dtype='float')
full_data_sim_median = np.array(np.median(full_data_sim_array, axis=0),
                                dtype='float')
full_data_sim_params = full_data['sim_params']

DelfiEnsemble = delfi.Delfi(full_data_sim_median, 
                            prior, NDEs,
                            param_limits = [lower, upper],
                            param_names = pn,
                            results_dir = RESULTS_DIR + '/' + str(base) + '/')

DelfiEnsemble.load_simulations(full_data_sim_array, full_data_sim_params)
DelfiEnsemble.train_ndes()

n_dim2d = nn
n_burn2d = 1000
n_steps2d = 10000
n_walkers2d = nn * n_dim2d

theta0_2d = np.array([list(initial_parameters(theta2d_expected_mean, 0.01))
                      for i in range(n_walkers2d)])

def prior_term2d(theta2d):
    p_ = theta2d.T
    for i in range(len(p_)):
        if (p_[i]<lower[i]) or (p_[i]>upper[i]):
            return -np.inf
    return 0.

def log_posterior2d_temp(theta2d, data):
    return DelfiEnsemble.log_posterior_stacked([theta2d.T],data=data)[0][0] + prior_term2d(theta2d)

sampler2d_ = mc.EnsembleSampler(n_walkers2d, n_dim2d,
            log_posterior2d_temp,args=(full_data['target'],))

_ = sampler2d_.run_mcmc(theta0_2d, n_burn2d+n_steps2d)
final_chain = sampler2d_.get_chain()

np.save(args.output + "_final_chain.npy", final_chain)
print(f"Saved chains to {args.output}_final_chain.npy.")

samples = MCSamples(samples=[final_chain[:,:,0].flatten(),
                             final_chain[:,:,1].flatten()],
                    names = ['Om','s8'],labels = [r'\Omega_{\rm m}',r'\sigma_8'],
                    label='chain',
                    settings={'mult_bias_correction_order':1,
                              'smooth_scale_2D':0.4,
                              'smooth_scale_1D':0.2})

cosmo = np.loadtxt(args.input_cosmo)
data_cosmo = cosmo[cosmo[:,0] == DATA_INDEX][0]
data_cosmo_Om, data_cosmo_s8 = data_cosmo[-2], data_cosmo[-1]

plt.figure(figsize=(10,10))
g = plots.get_subplot_plotter()

g.triangle_plot([samples],['Om','s8'],legend_loc='upper right',
                param_limits = {'Om': (0.2, 0.4),
                                's8': (0.65, 0.85)},
                markers = {'Om': data_cosmo_Om,
                           's8': data_cosmo_s8},
                marker_args={'lw': 2})
g.export(args.output + "_posteriors.png")

print(f"Saved contour plot to {args.output}_posteriors.png.")
