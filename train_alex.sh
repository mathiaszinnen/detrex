#!/bin/bash -l
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:a100:8
#SBATCH --job-name=focaldinotest
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python
module load cuda

source activate detrex

mkdir ${TMPDIR}/$SLURM_JOB_ID
cd ${TMPDIR}/$SLURM_JOB_ID

cp -r /home/woody/iwi5/iwi5064h/detrex .

cd detrex

pip install -e detectron2
pip install -e .

mkdir -p ./data/ODOR-v3
tar xf /home/janus/iwi5-datasets/odor3/odor3.tar -C ./data/ODOR-v3

export DETECTRON2_DATASETS=./data/ODOR-v3/

python tools/train_net.py --config-file projects/dino/configs/odor3_fn_l_lrf_384_fl4_5scale_50ep.py --num-gpus 8


echo "train done"
