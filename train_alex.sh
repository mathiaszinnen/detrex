#!/bin/bash -l
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:a40:1
#SBATCH --job-name=focaldinotest
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python

source activate detrex

mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r /home/woody/iwi5/iwi5064h/detrex .

cd detrex

tar xf /home/janus/iwi5-datasets/odor/odor3.tar

export DETECTRON2_DATASETS=./data/ODOR-v3/

python tools/train_net.py --config-file projects/dino/configs/odor3_fn_l_lrf_384_fl4_5scale_50ep.py


echo "train done"
