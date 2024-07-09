#!/bin/bash

#SBATCH --nodes=20
#SBATCH --time=23:59:59
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=40
#SBATCH --job-name=lensing_pipe
#SBATCH -o /home/s/sievers/kaper/scratch/lenspipe/output_sh/lensing_pipe_full_%j.out
#SBATCH -e /home/s/sievers/kaper/scratch/lenspipe/output_sh/lensing_pipe_full_%j.err
#SBATCH --mail-type=ALL

source ~/.bashrc
export ENLIB_COMP=niagara_gcc

cd ~/gitreps/DR6plus_lensing/exploration/mock_pipe/

export DISABLE_MPI=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

qid='pa4av4 pa5av4 pa5bv4 pa6av4 pa6bv4'
fp=/home/s/sievers/kaper/gitreps/CMB_lensing_SBI/code/lensing_pipeline/filepaths.yaml

#------
#MAKING MOCK DATA ####
#mpirun -n 1 --bind-to none python make_mock_init.py
#-----
#preprocessing
#mpirun -n 1 --bind-to none python downgrade.py --filepath ${fp} --qid ${qid} --prepare_maps --prepare_ivars --coadd
#mpirun -n 1 --bind-to none python new_inpaint.py --filepath ${fp} --qid ${qid} --prepare_maps --coadd 
#mpirun -n 1 --bind-to none python kspace_coadd.py --filepath ${fp} --qid ${qid} --recalculate_weights --split 0 #--model_subtract

#----
#sim making
#mpirun -n 20 python make_sims_if.py --filepath ${fp} --set 0 --qid ${qid} --coadd 
#mpirun -n 20 python make_sims_if.py --filepath ${fp} --set 1 --qid ${qid} --coadd

#----
#analysis
#mpirun -n 20 python filter.py --filepath ${fp}
#mpirun -n 1 --bind-to none python norms.py --filepath ${fp} --ph --bh --est1 MV
#mpirun -n 20 --bind-to none python mean_field.py --filepath ${fp} --ph --bh --est1 MV --set 0
#mpirun -n 20 --bind-to none python mean_field.py --filepath ${fp} --ph --bh --est1 MV --set 1
#mpirun -n 1 --bind-to none python auto.py --filepath ${fp} --ph --bh --est1 MV --nsets 2
#mpirun -n 1 --bind-to none python rdn0.py --filepath ${fp} --ph --bh --est1 MV --set 0
#mpirun -n 1 --bind-to none python rdn0.py --filepath ${fp} --ph --bh --est1 MV --set 1
#mpirun -n 1 --bind-to none python mcn1.py --filepath ${fp} --ph --bh --est MV 

